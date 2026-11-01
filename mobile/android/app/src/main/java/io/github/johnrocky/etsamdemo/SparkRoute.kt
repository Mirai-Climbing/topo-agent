package io.github.johnrocky.etsamdemo

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.Base64
import android.util.Log
import java.io.ByteArrayOutputStream
import java.net.URL
import javax.net.ssl.HttpsURLConnection
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.hypot
import kotlin.math.sin
import org.json.JSONArray
import org.json.JSONObject

/**
 * The server pipeline's Spark stage, ported for direct calls from the phone: per-hold colors,
 * groups and histogram hints, the prompt table, the annotated photo plus tap close-up, one
 * Muse Spark request, and strict answer parsing. No DINO embeddings on-device (the hist
 * column carries the similarity signal); the system prompt and table format mirror
 * `topo_agent/pipeline.py` (v3) exactly.
 */
object SparkRoute {
  const val API = "https://api.meta.ai/v1"
  const val MODEL = "muse-spark-1.3"
  const val EFFORT = "low"
  const val TEMP = 0.2
  const val AI_SIDE = 1280
  const val JPEG_QUALITY = 85
  const val SIMILAR = 0.12
  const val LINK = 0.3f
  const val MAX_HOLDS = 60
  const val HIST_TOP_K = 3
  const val TIMEOUT_MS = 120000

  const val SYSTEM_V3 =
      """You are a climbing route grouping engine, not an exploratory agent.

Task: find every hold of the same route as the tapped hold (marked * in the table and outlines).

Input: hold ids, positions (x, y, size on a 0-1000 scale), precomputed colors (names, HSV and
CIELAB features), color similarity groups, same-group spatial neighbors, the wall photo, a second
image where every hold is outlined and numbered, and a close-up of the tapped hold.

The table is a cheap color sensor, not the truth: on real photos chalk dust washes colors out and
shadows darken them, so color names and groups are often wrong. Trust your eyes over the table.

Instructions:
1. Answer in a single inference pass. Find ONLY the tapped hold's route, not the other routes.
2. First look at the close-up: a strip of tape next to the tapped hold means a tape route
   (holds of any color with the same tape; colors and groups do not help). Without tape,
   the route is the holds that look the same color as the tapped hold IN THE PHOTOS.
3. Decide the tapped hold's true color from the close-up and the photos, not from its table row.
   If the table calls it gray, white or black but it looks red, orange, yellow, green, blue or
   pink, the table is wrong: collect the holds that visually match the tap, whatever group the
   table puts them in.
4. Judge matches by hue family in the photos and ignore brightness: a hold in shadow looks darker
   and a hold under a lamp looks washed out, but the hue stays the same. Shadow or glare alone
   never excludes a hold.
5. Pale gray, white and beige holds that look like the bare wall or chalk dust are NOT the route
   when the tapped hold is a saturated color: exclude them even if the table puts them in the
   tapped hold's group. Pale holds qualify only when the tapped hold is itself pale, and then only
   the ones that clearly share its tint on the same wall.
6. One route is often split across several table groups (shadow turns green to gray or black,
   chalk turns red to pink or white). After the visual matches in the tapped hold's group, scan
   the OTHER groups for holds that look like the tapped hold in the photos on the same wall and
   line, and include them too.
7. The near column lists only direct neighbors; the route continues through chains (tap to A to B
   to C). Include visual matches reachable through a chain of same-looking holds, not just the
   tapped hold's direct neighbors.
8. A problem usually has 4 to 15 holds. Fewer than 4 means shadowed or far holds were missed:
   look again at the other groups and at far holds on the same line before answering.
9. The route runs from the bottom to the top as one continuous line you can follow from the tapped
   hold. A same-color hold across a gap or on another wall is usually another route: exclude it.
10. Never invent hold ids; use only the ids in the table. Always include the tapped hold.
11. When in doubt, compare the hold with the tapped hold in the photos: include it only if the
    colors clearly match and it sits on the same wall and line. Exclude wall-colored lookalikes
    and holds that are clearly a different color.
12. Return only valid JSON, without explanations or intermediate reasoning:
{"hold_ids": [1, 4, 7], "confidence": 0.95}"""

  data class HoldRow(
      val id: Int,
      val x: Float,
      val y: Float,
      val r: Float,
      val hsv: FloatArray,
      val lab: FloatArray,
      val colorName: String,
      val hist: FloatArray,
      var group: Int = 0,
      var near: List<Int> = emptyList(),
      var histTop: List<Pair<Int, Float>> = emptyList(),
  )

  data class Answer(val ids: List<Int>, val confidence: Double, val error: String)

  private val COLOR_NAMES =
      listOf(
          0.0f to "red", 0.05f to "orange", 0.12f to "yellow", 0.2f to "yellow-green",
          0.33f to "green", 0.47f to "cyan", 0.58f to "blue", 0.7f to "purple",
          0.83f to "pink", 0.93f to "red")

  private fun colored(hsv: FloatArray): Boolean = hsv[1] >= 0.28f && hsv[2] >= 0.15f

  fun colorName(hsv: FloatArray): String {
    if (!colored(hsv)) return if (hsv[2] < 0.25f) "black"
        else if (hsv[2] > 0.75f && hsv[1] < 0.15f) "white" else "gray"
    return COLOR_NAMES.filter { hsv[0] >= it.first }.last().second
  }

  private fun hueGap(a: Float, b: Float): Float = minOf(kotlin.math.abs(a - b), 1 - kotlin.math.abs(a - b))

  fun colorDistance(a: FloatArray, b: FloatArray): Float {
    if (colored(a) && colored(b))
        return hueGap(a[0], b[0]) * 2 + kotlin.math.abs(a[1] - b[1]) * 0.3f +
            kotlin.math.abs(a[2] - b[2]) * 0.2f
    if (!colored(a) && !colored(b))
        return kotlin.math.abs(a[2] - b[2]) + kotlin.math.abs(a[1] - b[1]) * 0.5f
    return 1.0f
  }

  private fun meanHsv(px: List<FloatArray>): FloatArray {
    var x = 0f
    var y = 0f
    for (c in px) {
      val a = c[0] * 2 * Math.PI.toFloat()
      x += cos(a) * c[1]
      y += sin(a) * c[1]
    }
    val h = ((atan2(y, x) / (2 * Math.PI.toFloat())) % 1 + 1) % 1
    return floatArrayOf(h, px.sumOf { it[1].toDouble() }.toFloat() / px.size,
        px.sumOf { it[2].toDouble() }.toFloat() / px.size)
  }

  /** sRGB (0-255) → CIELAB (D65), mirroring the server's rgb_to_lab. */
  fun rgbToLab(r: Float, g: Float, b: Float): FloatArray {
    fun lin(v: Float): Double {
      val u = v / 255.0
      return if (u <= 0.04045) u / 12.92 else Math.pow((u + 0.055) / 1.055, 2.4)
    }
    val rl = lin(r)
    val gl = lin(g)
    val bl = lin(b)
    val x = (rl * 0.4124 + gl * 0.3576 + bl * 0.1805) / 0.95047
    val yy = rl * 0.2126 + gl * 0.7152 + bl * 0.0722
    val z = (rl * 0.0193 + gl * 0.1192 + bl * 0.9505) / 1.08883
    fun f(t: Double): Double = if (t > 0.008856) Math.cbrt(t) else 7.787 * t + 16.0 / 116
    val fx = f(x)
    val fy = f(yy)
    val fz = f(z)
    return floatArrayOf((116 * fy - 16).toFloat(), (500 * (fx - fy)).toFloat(),
        (200 * (fy - fz)).toFloat())
  }

  /**
   * Per-hold colors, histograms, groups, neighbors and hint lists. Sampling mirrors
   * classify_colors: mask-interior pixels (stride 2); saturated holds average the hue of the
   * saturated pixels only, pale holds average the half farthest from the wall brightness.
   */
  fun classify(photo: Bitmap, dets: List<Det>, proto: FloatArray): List<HoldRow> {
    val start = System.nanoTime()
    val w = photo.width
    val h = photo.height
    val base = IntArray(w * h)
    photo.getPixels(base, 0, w, 0, 0, w, h)
    val scale = SIDE / maxOf(w, h).toFloat()
    val padX = (SIDE - w * scale) / 2f
    val padY = (SIDE - h * scale) / 2f
    val probe = FloatArray(3)
    val wallVs = mutableListOf<Float>()
    for (y in 0 until h step 6) {
      for (x in 0 until w step 6) {
        Color.colorToHSV(base[y * w + x], probe)
        wallVs.add(probe[2])
      }
    }
    val wallV = wallVs.sorted()[wallVs.size / 2]
    val rows =
        dets.mapIndexed { id, d ->
          val m = holdMask(d, w, h, scale, padX, padY, proto)
          val px = mutableListOf<FloatArray>()
          val rgb = mutableListOf<Int>()
          if (m != null) {
            for (y in 0 until m.bh step 2) {
              for (x in 0 until m.bw step 2) {
                if (!m.bin[y * m.bw + x]) continue
                val c = base[(m.by0 + y) * w + (m.bx0 + x)]
                Color.colorToHSV(c, probe)
                px.add(floatArrayOf(probe[0] / 360f, probe[1], probe[2]))
                rgb.add(c)
              }
            }
          }
          if (px.isEmpty()) { // degenerate mask: fall back to the box center pixel
            val cx = ((d.x1 + d.x2) / 2).toInt().coerceIn(0, w - 1)
            val cy = ((d.y1 + d.y2) / 2).toInt().coerceIn(0, h - 1)
            val c = base[cy * w + cx]
            Color.colorToHSV(c, probe)
            px.add(floatArrayOf(probe[0] / 360f, probe[1], probe[2]))
            rgb.add(c)
          }
          val sats = px.map { it[1] }.sorted()
          val medianS = sats[sats.size / 2]
          val selIdx =
              if (medianS >= 0.28f) {
                val cut = maxOf(0.2f, medianS * 0.5f)
                px.indices.filter { px[it][1] >= cut }.ifEmpty { px.indices.toList() }
              } else {
                px.indices.sortedByDescending { kotlin.math.abs(px[it][2] - wallV) }
                    .take(maxOf(1, px.size / 2))
              }
          val col = meanHsv(selIdx.map { px[it] })
          var mr = 0f
          var mg = 0f
          var mb = 0f
          for (i in selIdx) {
            mr += (rgb[i] shr 16 and 0xFF)
            mg += (rgb[i] shr 8 and 0xFF)
            mb += (rgb[i] and 0xFF)
          }
          val n = selIdx.size.toFloat()
          val lab = rgbToLab(mr / n, mg / n, mb / n)
          val hist = FloatArray(HIST_H * HIST_S)
          for (c in px) {
            val hi = (c[0] * HIST_H).toInt().coerceIn(0, HIST_H - 1)
            val si = (c[1] * HIST_S).toInt().coerceIn(0, HIST_S - 1)
            hist[hi * HIST_S + si] += 1f
          }
          for (i in hist.indices) hist[i] /= px.size
          HoldRow(
              id = id,
              x = (d.x1 + d.x2) / 2 / w,
              y = (d.y1 + d.y2) / 2 / h,
              r = maxOf(d.x2 - d.x1, d.y2 - d.y1) / 2 / w,
              hsv = floatArrayOf((col[0] * 1000).toInt() / 1000f, (col[1] * 1000).toInt() / 1000f,
                  (col[2] * 1000).toInt() / 1000f),
              lab = floatArrayOf((lab[0] * 10).toInt() / 10f, (lab[1] * 10).toInt() / 10f,
                  (lab[2] * 10).toInt() / 10f),
              colorName = colorName(col),
              hist = hist,
          )
        }
    groupByColor(rows)
    addNear(rows, w, h)
    addHistTop(rows)
    Log.i(TAG, "spark classify %.0f ms for %d holds".format((System.nanoTime() - start) / 1e6, rows.size))
    return rows
  }

  private fun groupByColor(rows: List<HoldRow>) {
    val parent = rows.map { it.id to it.id }.toMap().toMutableMap()
    fun root(i: Int): Int {
      var j = i
      while (parent[j] != j) {
        parent[j] = parent[parent[j]!!]!!
        j = parent[j]!!
      }
      return j
    }
    for (k in rows.indices) {
      for (l in k + 1 until rows.size) {
        if (colorDistance(rows[k].hsv, rows[l].hsv) <= SIMILAR)
            parent[root(rows[k].id)] = root(rows[l].id)
      }
    }
    val labels = mutableMapOf<Int, Int>()
    for (c in rows) {
      val r = root(c.id)
      labels.putIfAbsent(r, labels.size)
      c.group = labels[r]!!
    }
  }

  private fun addNear(rows: List<HoldRow>, width: Int, height: Int) {
    val side = maxOf(width, height).toFloat()
    for (a in rows) {
      a.near =
          rows.filter {
                it.id != a.id && it.group == a.group &&
                    hypot((a.x - it.x) * width, (a.y - it.y) * height) <= LINK * side
              }
              .map { it.id }
              .sorted()
    }
  }

  internal fun histSim(a: FloatArray, b: FloatArray): Float {
    var s = 0f
    for (i in a.indices) s += kotlin.math.sqrt(a[i] * b[i])
    return s.coerceIn(0f, 1f)
  }

  private fun addHistTop(rows: List<HoldRow>) {
    for (a in rows) {
      a.histTop =
          rows.filter { it.id != a.id }
              .map { it.id to histSim(a.hist, it.hist) }
              .sortedBy { it.second }
              .takeLast(HIST_TOP_K)
              .reversed()
    }
  }

  /** The prompt table, mirroring prompt_table (hist column, no embed column on-device). */
  fun buildTable(rows: List<HoldRow>, tapId: Int): String {
    val sb = StringBuilder("id | color | hue sat val | L a b | group | x y size | near | hist\n")
    for (c in rows) {
      val mark = if (c.id == tapId) "*" else ""
      sb.append(
          "%d%s | %s | %.3f %.3f %.3f | %.1f %.1f %.1f | %d | %d %d %d | %s | %s\n".format(
              java.util.Locale.US, c.id, mark, c.colorName, c.hsv[0], c.hsv[1], c.hsv[2],
              c.lab[0], c.lab[1], c.lab[2], c.group, (c.x * 1000).toInt(), (c.y * 1000).toInt(),
              (c.r * 1000).toInt(), c.near.joinToString(","),
              c.histTop.joinToString(",") { "%d:%.2f".format(java.util.Locale.US, it.first, it.second) }))
    }
    val groups = rows.groupBy { it.group }.toSortedMap()
    val detail =
        groups.entries.joinToString("; ") { (g, cs) ->
          "%d (%s): [%s]".format(g, cs.first().colorName, cs.map { it.id }.joinToString(", "))
        }
    sb.append("Color groups: %s\n".format(detail))
    sb.append(
        "Hist lists the holds whose color distribution best matches each row " +
            "as id:score (higher is more similar); same-route holds usually rank " +
            "each other high even across shadow and glare.\n")
    sb.append(
        "The user tapped hold %d (marked %d*). Return ONLY the holds of the tapped hold's route as JSON."
            .format(tapId, tapId))
    return sb.toString()
  }

  /** Every hold boxed and numbered (tap magenta, others yellow), like annotate. */
  fun annotate(photo: Bitmap, dets: List<Det>, tapId: Int): Bitmap {
    val s = AI_SIDE / maxOf(photo.width, photo.height).toFloat()
    val out =
        Bitmap.createScaledBitmap(photo, (photo.width * s).toInt(), (photo.height * s).toInt(), true)
            .copy(Bitmap.Config.ARGB_8888, true)
    val canvas = Canvas(out)
    val w = out.width.toFloat()
    val h = out.height.toFloat()
    val boxPaint = Paint().apply { style = Paint.Style.STROKE }
    val textPaint = Paint().apply {
      color = Color.WHITE
      textSize = maxOf(14f, out.width / 70f)
      textAlign = Paint.Align.CENTER
    }
    val bgPaint = Paint().apply { color = Color.BLACK }
    for ((id, d) in dets.withIndex()) {
      boxPaint.color = if (id == tapId) Color.MAGENTA else Color.YELLOW
      boxPaint.strokeWidth = if (id == tapId) 5f else 2f
      canvas.drawRect(d.x1 * s, d.y1 * s, d.x2 * s, d.y2 * s, boxPaint)
    }
    for ((id, d) in dets.withIndex()) {
      val label = if (id == tapId) "%d*".format(id) else "%d".format(id)
      val x = (d.x1 + d.x2) / 2 * s
      val y = (d.y1 + d.y2) / 2 * s
      val tw = textPaint.measureText(label)
      val th = textPaint.textSize
      canvas.drawRect(x - tw / 2 - 2, y - th - 1, x + tw / 2 + 2, y + 1, bgPaint)
      canvas.drawText(label, x, y, textPaint)
    }
    return out
  }

  /** 256px tap close-up tile, mirroring Tools.look_closer for one hold. */
  fun closeUp(photo: Bitmap, d: Det, id: Int): Bitmap {
    val w = photo.width
    val h = photo.height
    val r = maxOf(d.x2 - d.x1, d.y2 - d.y1) / 2 / w
    val half = maxOf(r * 2.5f, 0.04f) * w
    val cx = (d.x1 + d.x2) / 2
    val cy = (d.y1 + d.y2) / 2
    val x0 = (cx - half).toInt().coerceIn(0, w - 1)
    val y0 = (cy - half).toInt().coerceIn(0, h - 1)
    val x1 = (cx + half).toInt().coerceIn(x0 + 1, w)
    val y1 = (cy + half).toInt().coerceIn(y0 + 1, h)
    val tile =
        Bitmap.createScaledBitmap(
            Bitmap.createBitmap(photo, x0, y0, x1 - x0, y1 - y0), 256, 256, true)
            .copy(Bitmap.Config.ARGB_8888, true)
    val paint = Paint().apply {
      color = Color.WHITE
      textSize = 24f
      style = Paint.Style.FILL_AND_STROKE
      strokeWidth = 3f
    }
    Canvas(tile).drawText("%d".format(id), 6f, 28f, paint)
    return tile
  }

  fun smallPhoto(photo: Bitmap): Bitmap {
    val s = (AI_SIDE / maxOf(photo.width, photo.height).toFloat()).coerceAtMost(1f)
    return Bitmap.createScaledBitmap(
        photo, (photo.width * s).toInt(), (photo.height * s).toInt(), true)
  }

  fun jpegBytes(bmp: Bitmap): ByteArray {
    val buf = ByteArrayOutputStream()
    bmp.compress(Bitmap.CompressFormat.JPEG, JPEG_QUALITY, buf)
    return buf.toByteArray()
  }

  private fun imagePart(jpeg: ByteArray): JSONObject =
      JSONObject()
          .put("type", "image_url")
          .put("image_url", JSONObject().put(
              "url", "data:image/jpeg;base64," + Base64.encodeToString(jpeg, Base64.NO_WRAP)))

  /** The single-pass request body; image parts mirror run_once (photo, annotated, close-up). */
  fun requestBody(table: String, small: ByteArray, annotated: ByteArray, closeUp: ByteArray): String =
      JSONObject()
          .put("model", MODEL)
          .put("messages", JSONArray()
              .put(JSONObject().put("role", "system").put("content", SYSTEM_V3))
              .put(JSONObject().put("role", "user").put("content", JSONArray()
                  .put(JSONObject().put("type", "text").put("text", table))
                  .put(imagePart(small))
                  .put(imagePart(annotated))
                  .put(imagePart(closeUp)))))
          .put("temperature", TEMP)
          .put("store", false)
          .put("reasoning_effort", EFFORT)
          .toString()

  /** POST the body; returns choices[0].message.content. Throws on transport or API errors. */
  fun askSpark(key: String, body: String): String {
    val conn = URL("$API/chat/completions").openConnection() as HttpsURLConnection
    try {
      conn.requestMethod = "POST"
      conn.setRequestProperty("Authorization", "Bearer $key")
      conn.setRequestProperty("Content-Type", "application/json")
      conn.connectTimeout = TIMEOUT_MS
      conn.readTimeout = TIMEOUT_MS
      conn.doOutput = true
      conn.outputStream.use { it.write(body.toByteArray(Charsets.UTF_8)) }
      val code = conn.responseCode
      val raw =
          (if (code < 400) conn.inputStream else conn.errorStream)?.readBytes()?.toString(Charsets.UTF_8)
              ?: ""
      if (code >= 400) throw RuntimeException("Muse Spark $code: ${raw.take(300)}")
      val msg = JSONObject(raw).getJSONArray("choices").getJSONObject(0).getJSONObject("message")
      return msg.optString("content", "")
    } finally {
      conn.disconnect()
    }
  }

  /** Route JSON → Answer, mirroring parse_answer + clean_ids (tap first, known unique ids). */
  fun parseAnswer(text: String, known: Set<Int>, tapId: Int): Answer {
    var body = text.trim()
    if (body.startsWith("```")) {
      body = if ("\n" in body) body.split("\n", limit = 2)[1] else ""
      body = body.substringBeforeLast("```")
    }
    val data = try {
      JSONObject(body)
    } catch (e: Exception) {
      return Answer(emptyList(), 0.0, "not valid JSON")
    }
    val ids = data.optJSONArray("hold_ids") ?: return Answer(emptyList(), 0.0, "missing hold_ids")
    val conf = data.optDouble("confidence", Double.NaN)
    if (ids.length() == 0 || conf.isNaN() || conf < 0 || conf > 1)
        return Answer(emptyList(), 0.0, "missing hold_ids or confidence")
    val clean = mutableListOf<Int>()
    for (i in 0 until ids.length()) {
      val v = ids.opt(i)
      if (v !is Int && v !is Long) return Answer(emptyList(), 0.0, "unknown id $v")
      val id = (v as Number).toInt()
      if (id !in known) return Answer(emptyList(), 0.0, "unknown id $id")
      if (id in clean) return Answer(emptyList(), 0.0, "hold $id is assigned twice")
      clean.add(id)
    }
    val out = mutableListOf(tapId)
    for (i in clean) if (i !in out) out.add(i)
    return Answer(out.take(MAX_HOLDS), conf, "")
  }

  // --- mask math (shared with MainActivity rendering) ---

  data class MaskBox(val bx0: Int, val by0: Int, val bw: Int, val bh: Int, val bin: BooleanArray)

  /** Binary hold mask at photo resolution over the detection box, or null for empty boxes. */
  fun holdMask(
      d: Det, w: Int, h: Int, scale: Float, padX: Float, padY: Float, proto: FloatArray
  ): MaskBox? {
    val gx0 = (d.x1 * scale + padX) / 4
    val gy0 = (d.y1 * scale + padY) / 4
    val gx1 = (d.x2 * scale + padX) / 4
    val gy1 = (d.y2 * scale + padY) / 4
    val ix0 = (gx0.toInt() - 1).coerceIn(0, PROTO - 1)
    val iy0 = (gy0.toInt() - 1).coerceIn(0, PROTO - 1)
    val ix1 = (gx1.toInt() + 1).coerceIn(0, PROTO - 1)
    val iy1 = (gy1.toInt() + 1).coerceIn(0, PROTO - 1)
    val mw = ix1 - ix0 + 1
    val mh = iy1 - iy0 + 1
    val small = FloatArray(mw * mh)
    for (cy in 0 until mh) {
      for (cx in 0 until mw) {
        val p = (iy0 + cy) * PROTO + (ix0 + cx)
        var v = 0f
        for (k in 0 until COEFS) v += d.coefs[k] * proto[k * PROTO * PROTO + p]
        small[cy * mw + cx] = 1f / (1f + exp(-v))
      }
    }
    val bx0 = d.x1.toInt().coerceIn(0, w - 1)
    val by0 = d.y1.toInt().coerceIn(0, h - 1)
    val bx1 = d.x2.toInt().coerceIn(0, w)
    val by1 = d.y2.toInt().coerceIn(0, h)
    val bw = bx1 - bx0
    val bh = by1 - by0
    if (bw <= 0 || bh <= 0) return null
    val bin = BooleanArray(bw * bh)
    for (y in 0 until bh) {
      val fy = ((by0 + y + 0.5f) * scale + padY) / 4 - iy0
      val y0 = fy.toInt().coerceIn(0, mh - 1)
      val y1 = (y0 + 1).coerceIn(0, mh - 1)
      val wy = (fy - fy.toInt()).coerceIn(0f, 1f)
      for (x in 0 until bw) {
        val fx = ((bx0 + x + 0.5f) * scale + padX) / 4 - ix0
        val x0 = fx.toInt().coerceIn(0, mw - 1)
        val x1 = (x0 + 1).coerceIn(0, mw - 1)
        val wx = (fx - fx.toInt()).coerceIn(0f, 1f)
        val v =
            small[y0 * mw + x0] * (1 - wx) * (1 - wy) +
                small[y0 * mw + x1] * wx * (1 - wy) +
                small[y1 * mw + x0] * (1 - wx) * wy +
                small[y1 * mw + x1] * wx * wy
        bin[y * bw + x] = v > 0.5f
      }
    }
    return MaskBox(bx0, by0, bw, bh, bin)
  }
}
