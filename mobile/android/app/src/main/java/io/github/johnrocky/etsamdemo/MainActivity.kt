package io.github.johnrocky.etsamdemo

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.Typeface
import android.media.ExifInterface
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import android.util.Log
import android.view.MotionEvent
import android.view.View
import android.text.InputType
import android.view.ViewGroup
import android.widget.Button
import android.widget.EditText
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.FileProvider
import java.io.File
import kotlin.concurrent.thread
import org.pytorch.executorch.EValue
import org.pytorch.executorch.Module
import org.pytorch.executorch.Tensor

internal const val TAG = "ETSAMDemo"

private const val BACKGROUND = 0xFF101014.toInt()
private const val ACCENT = 0xFF6FE3B4.toInt()
private const val MUTED = 0xFF9AA0A6.toInt()

internal const val SIDE = 640
internal const val PROTO = 160
internal const val COEFS = 32
private const val ANCHORS = 8400
private const val CONF_THRESH = 0.5f
private const val NMS_THRESH = 0.5f
private const val MAX_DETS = 100
/** Detection and overlay run on this; the phone screen cannot show more anyway. */
private const val MAX_SIDE = 1400
/** Hue/saturation bins of the per-hold color histogram used for route grouping. */
internal const val HIST_H = 16
internal const val HIST_S = 16
/** Holds at least this similar (Bhattacharyya) to the tapped hold count as its route. */
private const val ROUTE_SIM = 0.6f

data class Det(
    val x1: Float,
    val y1: Float,
    val x2: Float,
    val y2: Float,
    val conf: Float,
    val coefs: FloatArray,
)

/**
 * One-shot climbing-hold detector. A YOLO11n-seg model trained on the Kaggle hold dataset runs
 * once per photo; nothing is drawn until a tap. Tapping a hold groups its route offline:
 * the route keeps color while everything else goes grayscale, with a target frame on the tap.
 * Tapping the wall clears the route. With a Muse Spark API key saved in settings, each tap also
 * asks Spark (one call, same prompt as the server pipeline) and the Spark answer replaces the
 * offline grouping; without a key or on any failure the offline grouping stands.
 *
 * Model contract (ExecuTorch export of the ultralytics seg head):
 * - input [1,3,640,640] letterboxed RGB / 255, pad value 114
 * - output 0 [1,37,8400]: rows 0-3 cx/cy/w/h, row 4 confidence, rows 5-36 mask coefficients
 * - output 1 [1,32,160,160]: mask prototypes; mask = sigmoid(coefs . proto)
 */
class MainActivity : AppCompatActivity() {

  private lateinit var view: ImageView
  private lateinit var status: TextView
  private var photo: Bitmap = Bitmap.createBitmap(1, 1, Bitmap.Config.ARGB_8888)
  private var hasPhoto = false
  private var dets: List<Det> = emptyList()
  private var selected = -1
  private var hists: List<FloatArray> = emptyList()
  private var route: Set<Int> = emptySet()
  private var rows: List<SparkRoute.HoldRow> = emptyList()
  @Volatile private var sparkBusy = false
  private lateinit var timerView: TextView
  private val timerHandler = Handler(Looper.getMainLooper())
  private var sparkStartMs = 0L
  private val timerTick =
      object : Runnable {
        override fun run() {
          if (!sparkBusy) return
          timerView.text =
              "asking spark… %.1f s".format((SystemClock.elapsedRealtime() - sparkStartMs) / 1000.0)
          timerHandler.postDelayed(this, 100)
        }
      }
  private var captureFile = File("")
  @Volatile private var busy = true

  private val cameraLauncher =
      registerForActivityResult(ActivityResultContracts.TakePicture()) { ok ->
        if (ok) onCapture()
        else setStatus("capture cancelled · %d holds on current photo".format(dets.size))
      }

  private val galleryLauncher =
      registerForActivityResult(ActivityResultContracts.GetContent()) { uri ->
        if (uri != null) onGalleryImage(uri)
        else setStatus("pick cancelled · %d holds on current photo".format(dets.size))
      }

  override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(savedInstanceState)
    buildUi()
    thread {
      android.os.Process.setThreadPriority(android.os.Process.THREAD_PRIORITY_DISPLAY)
      runCatching { load() }.onFailure {
        Log.e(TAG, "load failed", it)
        setStatus("failed: ${it.message}")
      }
    }
  }

  private fun preprocess(bitmap: Bitmap): Pair<Tensor, FloatArray> {
    val scale = SIDE / maxOf(bitmap.width, bitmap.height).toFloat()
    val rw = (bitmap.width * scale).toInt()
    val rh = (bitmap.height * scale).toInt()
    val scaled = Bitmap.createScaledBitmap(bitmap, rw, rh, true)
    val px = IntArray(rw * rh)
    scaled.getPixels(px, 0, rw, 0, 0, rw, rh)
    val padX = (SIDE - rw) / 2
    val padY = (SIDE - rh) / 2
    val data = FloatArray(3 * SIDE * SIDE) { 114f / 255f }
    for (y in 0 until rh) {
      for (x in 0 until rw) {
        val c = px[y * rw + x]
        val dx = padX + x
        val dy = padY + y
        data[dy * SIDE + dx] = (c shr 16 and 0xFF) / 255f
        data[SIDE * SIDE + dy * SIDE + dx] = (c shr 8 and 0xFF) / 255f
        data[2 * SIDE * SIDE + dy * SIDE + dx] = (c and 0xFF) / 255f
      }
    }
    val tensor = Tensor.fromBlob(data, longArrayOf(1, 3, SIDE.toLong(), SIDE.toLong()))
    return tensor to floatArrayOf(scale, padX.toFloat(), padY.toFloat())
  }

  private fun nms(boxes: List<Det>): List<Det> {
    val sorted = boxes.sortedByDescending { it.conf }.toMutableList()
    val kept = mutableListOf<Det>()
    while (sorted.isNotEmpty() && kept.size < MAX_DETS) {
      val a = sorted.removeAt(0)
      kept.add(a)
      val it = sorted.iterator()
      while (it.hasNext()) {
        val b = it.next()
        val ix1 = maxOf(a.x1, b.x1)
        val iy1 = maxOf(a.y1, b.y1)
        val ix2 = minOf(a.x2, b.x2)
        val iy2 = minOf(a.y2, b.y2)
        val iw = maxOf(0f, ix2 - ix1)
        val ih = maxOf(0f, iy2 - iy1)
        val inter = iw * ih
        val union = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter
        if (union > 0 && inter / union > NMS_THRESH) it.remove()
      }
    }
    return kept
  }

  private fun modelFile(): File? {
    val dir = getExternalFilesDir(null) ?: return null
    return File(dir, "hold_yolo.pte").takeIf { it.exists() }
  }

  private fun load() {
    val model = modelFile()
    if (model == null) {
      setStatus("push hold_yolo.pte to the app files dir")
      busy = false
      return
    }
    val sample = runCatching { fitBitmap(loadAsset("gym.jpg")) }.getOrNull()
    if (sample == null) {
      setStatus("no sample photo — tap Camera to capture a wall")
      busy = false
      return
    }
    photo = sample
    hasPhoto = true
    runOnUiThread { view.setImageBitmap(photo) }
    detect(model, "sample")
  }

  private fun fitBitmap(raw: Bitmap): Bitmap {
    val s = (MAX_SIDE / maxOf(raw.width, raw.height).toFloat()).coerceAtMost(1f)
    return Bitmap.createScaledBitmap(raw, (raw.width * s).toInt(), (raw.height * s).toInt(), true)
  }

  private fun openCamera() {
    if (busy) return
    if (modelFile() == null) {
      setStatus("push hold_yolo.pte to the app files dir")
      return
    }
    val dir = File(cacheDir, "captures").apply { mkdirs() }
    captureFile = File(dir, "capture.jpg")
    val uri = FileProvider.getUriForFile(this, "$packageName.provider", captureFile)
    cameraLauncher.launch(uri)
  }

  private fun openGallery() {
    if (busy) return
    if (modelFile() == null) {
      setStatus("push hold_yolo.pte to the app files dir")
      return
    }
    galleryLauncher.launch("image/*")
  }

  private fun onGalleryImage(uri: android.net.Uri) {
    thread {
      runCatching {
        busy = true
        val raw =
            contentResolver.openInputStream(uri)?.use { BitmapFactory.decodeStream(it) }
                ?: error("cannot decode picked image")
        val orient =
            contentResolver.openInputStream(uri)?.use {
              ExifInterface(it).getAttributeInt(
                  ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL)
            } ?: ExifInterface.ORIENTATION_NORMAL
        useExternalPhoto(raw, orient, "gallery")
      }.onFailure {
        Log.e(TAG, "gallery failed", it)
        setStatus("gallery failed: ${it.message}")
        busy = false
      }
    }
  }

  /** Camera photos carry EXIF rotation; apply it before anything else. */
  private fun onCapture() {
    thread {
      runCatching {
        busy = true
        val raw = BitmapFactory.decodeFile(captureFile.absolutePath) ?: error("cannot decode capture")
        val orient =
            ExifInterface(captureFile.absolutePath)
                .getAttributeInt(
                    ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL)
        useExternalPhoto(raw, orient, "camera")
      }.onFailure {
        Log.e(TAG, "capture failed", it)
        setStatus("capture failed: ${it.message}")
        busy = false
      }
    }
  }

  private fun useExternalPhoto(raw: Bitmap, orient: Int, source: String) {
    val matrix = Matrix()
    when (orient) {
      ExifInterface.ORIENTATION_ROTATE_90 -> matrix.postRotate(90f)
      ExifInterface.ORIENTATION_ROTATE_180 -> matrix.postRotate(180f)
      ExifInterface.ORIENTATION_ROTATE_270 -> matrix.postRotate(270f)
    }
    val fixed =
        if (orient == ExifInterface.ORIENTATION_NORMAL) raw
        else Bitmap.createBitmap(raw, 0, 0, raw.width, raw.height, matrix, true)
    photo = fitBitmap(fixed)
    hasPhoto = true
    runOnUiThread { view.setImageBitmap(photo) }
    detect(modelFile() ?: error("model gone"), source)
  }

  private fun detect(model: File, source: String) {
    busy = true
    setStatus("detecting holds…")
    val (input, geo) = preprocess(photo)
    val scale = geo[0]
    val padX = geo[1]
    val padY = geo[2]
    val module = Module.load(model.absolutePath)
    val start = System.nanoTime()
    val outputs = module.forward(EValue.from(input))
    val inferMs = (System.nanoTime() - start) / 1e6
    val det = outputs[0].toTensor().dataAsFloatArray
    protoCache = outputs[1].toTensor().dataAsFloatArray
    module.destroy()

    // Output 0 is [37,8400]: candidate i reads column i across the 37 rows.
    val cands = mutableListOf<Det>()
    for (i in 0 until ANCHORS) {
      val conf = det[4 * ANCHORS + i]
      if (conf < CONF_THRESH) continue
      val cx = det[i]
      val cy = det[ANCHORS + i]
      val w = det[2 * ANCHORS + i]
      val h = det[3 * ANCHORS + i]
      // Back to photo pixels through the letterbox.
      val x1 = ((cx - w / 2 - padX) / scale).coerceIn(0f, photo.width.toFloat())
      val y1 = ((cy - h / 2 - padY) / scale).coerceIn(0f, photo.height.toFloat())
      val x2 = ((cx + w / 2 - padX) / scale).coerceIn(0f, photo.width.toFloat())
      val y2 = ((cy + h / 2 - padY) / scale).coerceIn(0f, photo.height.toFloat())
      if (x2 - x1 < 2 || y2 - y1 < 2) continue
      val coefs = FloatArray(COEFS) { k -> det[(5 + k) * ANCHORS + i] }
      cands.add(Det(x1, y1, x2, y2, conf, coefs))
    }
    dets = nms(cands)
    selected = -1
    route = emptySet()
    rows = SparkRoute.classify(photo, dets, protoCache)
    hists = rows.map { it.hist }
    Log.i(TAG, "yolo: ${dets.size} holds after NMS (${cands.size} candidates), infer ${inferMs}ms")
    render()
    setStatus(
        "hold-yolo · %s · %d holds · infer %.0f ms · tap a hold"
            .format(source, dets.size, inferMs))
    busy = false
  }

  private var overlayBmp: Bitmap? = null
  private var showOverlay = true
  private lateinit var toggleBtn: Button

  private fun updateView() {
    val bmp = if (showOverlay) overlayBmp ?: photo else photo
    runOnUiThread {
      view.setImageBitmap(bmp)
      toggleBtn.text = if (showOverlay) "Show original" else "Show detection"
    }
  }

  /**
   * Holds are never tinted. With no selection the photo shows as-is; after a tap everything
   * outside the tapped hold's route goes grayscale and only the route keeps color, whether the
   * route came from offline grouping or Spark. The tapped hold always gets a black-and-white
   * target frame.
   */
  private fun render() {
    val start = System.nanoTime()
    val w = photo.width
    val h = photo.height
    val base = IntArray(w * h)
    photo.getPixels(base, 0, w, 0, 0, w, h)
    val out = base.copyOf()
    val grouped = selected >= 0 && route.isNotEmpty()
    if (grouped) {
      val scale = SIDE / maxOf(w, h).toFloat()
      val padX = (SIDE - w * scale) / 2f
      val padY = (SIDE - h * scale) / 2f
      val cover = BooleanArray(w * h)
      for (idx in route) {
        val m =
            if (idx in dets.indices)
                SparkRoute.holdMask(dets[idx], w, h, scale, padX, padY, protoCache)
            else null
        if (m == null) continue
        for (y in 0 until m.bh) {
          for (x in 0 until m.bw) {
            if (m.bin[y * m.bw + x]) cover[(m.by0 + y) * w + (m.bx0 + x)] = true
          }
        }
      }
      for (o in out.indices) {
        if (cover[o]) continue
        val c = out[o]
        val lum =
            ((c shr 16 and 0xFF) * 0.299f + (c shr 8 and 0xFF) * 0.587f + (c and 0xFF) * 0.114f)
                .toInt()
        out[o] = Color.rgb(lum, lum, lum)
      }
    }
    Log.i(TAG, "render %.0f ms for %d masks".format((System.nanoTime() - start) / 1e6, dets.size))
    overlayBmp = SparkRoute.newOverlay(out, w, h)
    if (selected >= 0 && selected < dets.size) drawTapFrame(overlayBmp!!, dets[selected])
    showOverlay = true
    updateView()
  }

  /** Black-and-white target frame around the tapped hold's box: visible on any wall. */
  internal fun drawTapFrame(bmp: Bitmap, d: Det) {
    val canvas = Canvas(bmp)
    val pad = 8f
    val outer = Paint().apply {
      style = Paint.Style.STROKE
      strokeWidth = 10f
      color = Color.BLACK
    }
    val inner = Paint().apply {
      style = Paint.Style.STROKE
      strokeWidth = 5f
      color = Color.WHITE
    }
    canvas.drawRect(d.x1 - pad, d.y1 - pad, d.x2 + pad, d.y2 + pad, outer)
    canvas.drawRect(d.x1 - pad, d.y1 - pad, d.x2 + pad, d.y2 + pad, inner)
  }

  private var protoCache: FloatArray = FloatArray(0)

  private fun select(x: Float, y: Float) {
    if (busy || dets.isEmpty() || !hasPhoto) return
    // Smallest containing box wins: taps land on holds, not on the wall behind them.
    var best = -1
    var bestArea = Float.MAX_VALUE
    for ((i, d) in dets.withIndex()) {
      if (x >= d.x1 && x <= d.x2 && y >= d.y1 && y <= d.y2) {
        val a = (d.x2 - d.x1) * (d.y2 - d.y1)
        if (a < bestArea) {
          bestArea = a
          best = i
        }
      }
    }
    if (best < 0) { // tapping the wall clears the route
      if (selected >= 0) {
        selected = -1
        route = emptySet()
            render()
        setStatus("%d holds · tap a hold".format(dets.size))
      }
      return
    }
    selected = best
    route =
        if (hists.size == dets.size)
            dets.indices.filter { SparkRoute.histSim(hists[best], hists[it]) >= ROUTE_SIM }.toSet()
        else setOf(best)
    val d = dets[best]
    render()
    setStatus("hold #%d · route %d/%d holds · conf %.2f".format(best, route.size, dets.size, d.conf))
    Log.i(TAG, "tap ($x, $y) -> hold #$best route ${route.size}")
    askSpark(best)
  }

  private fun sparkKey(): String =
      getSharedPreferences("topo", MODE_PRIVATE).getString("spark_key", "")?.trim() ?: ""

  /** One Spark call for the tapped route; the answer replaces the offline grouping. */
  private fun askSpark(tapId: Int) {
    val key = sparkKey()
    if (key.isEmpty() || sparkBusy) return
    if (rows.size != dets.size || tapId !in dets.indices) return
    sparkBusy = true
    setStatus("hold #%d · asking spark…".format(tapId))
    sparkStartMs = SystemClock.elapsedRealtime()
    timerView.visibility = View.VISIBLE
    timerHandler.post(timerTick)
    val snapPhoto = photo
    val snapDets = dets.toList()
    val snapRows = rows
    thread {
      try {
        val table = SparkRoute.buildTable(snapRows, tapId)
        val small = SparkRoute.jpegBytes(SparkRoute.smallPhoto(snapPhoto))
        val annotated = SparkRoute.jpegBytes(SparkRoute.annotate(snapPhoto, snapDets, tapId))
        val closeUp = SparkRoute.jpegBytes(SparkRoute.closeUp(snapPhoto, snapDets[tapId], tapId))
        val start = System.nanoTime()
        val body = SparkRoute.requestBody(table, small, annotated, closeUp)
        Log.i(TAG, "spark upload %.0f KB".format(body.length / 1024.0))
        val text = SparkRoute.askSpark(key, body)
        val ms = (System.nanoTime() - start) / 1e6
        val ans = SparkRoute.parseAnswer(text, snapDets.indices.toSet(), tapId)
        runOnUiThread {
          sparkBusy = false
          timerHandler.removeCallbacks(timerTick)
          timerView.visibility = View.GONE
          if (selected != tapId || dets.size != snapDets.size) return@runOnUiThread
          if (ans.error.isNotEmpty()) {
            setStatus("hold #%d · spark unclear (%s), offline route kept".format(tapId, ans.error))
            return@runOnUiThread
          }
          route = ans.ids.toSet()
          render()
          setStatus("hold #%d · spark route %d/%d · conf %.2f · %.0f ms"
              .format(tapId, route.size, dets.size, ans.confidence, ms))
          Log.i(TAG, "spark tap #$tapId -> ${route.size} holds conf ${ans.confidence} ${ms}ms")
        }
      } catch (e: Exception) {
        Log.w(TAG, "spark failed", e)
        runOnUiThread {
          sparkBusy = false
          timerHandler.removeCallbacks(timerTick)
          timerView.visibility = View.GONE
          if (selected == tapId)
              setStatus("hold #%d · spark failed (%s), offline route kept"
                  .format(tapId, (e.message ?: "error").take(60)))
        }
      }
    }
  }

  private fun buildUi() {
    val root =
        LinearLayout(this).apply {
          orientation = LinearLayout.VERTICAL
          setBackgroundColor(BACKGROUND)
          setPadding(dp(14), dp(44), dp(14), dp(14))
        }
    supportActionBar?.hide()
    root.addView(
        TextView(this).apply {
          text = "Hold detector"
          setTextColor(Color.WHITE)
          textSize = 21f
          setTypeface(typeface, Typeface.BOLD)
        })
    root.addView(
        TextView(this).apply {
          text = "YOLO11s-seg · one pass, every hold · tap to group the route"
          setTextColor(MUTED)
          textSize = 13f
          setPadding(0, dp(2), 0, dp(4))
        })
    status =
        TextView(this).apply {
          text = "starting…"
          setTextColor(ACCENT)
          textSize = 15f
          setTypeface(typeface, Typeface.BOLD)
          setPadding(0, 0, 0, dp(10))
        }
    root.addView(status)
    timerView =
        TextView(this).apply {
          text = "asking spark… 0.0 s"
          setTextColor(Color.WHITE)
          textSize = 22f
          setTypeface(typeface, Typeface.BOLD)
          setPadding(0, 0, 0, dp(8))
          visibility = View.GONE
        }
    root.addView(timerView)
    root.addView(
        Button(this).apply {
          text = "Camera"
          setOnClickListener { openCamera() }
        })
    root.addView(
        Button(this).apply {
          text = "Gallery"
          setOnClickListener { openGallery() }
        })
    toggleBtn =
        Button(this).apply {
          text = "Show original"
          setOnClickListener {
            showOverlay = !showOverlay
            updateView()
          }
        }
    root.addView(toggleBtn)
    val keyRow =
        LinearLayout(this).apply {
          orientation = LinearLayout.HORIZONTAL
          setPadding(0, dp(6), 0, dp(6))
        }
    val keyField =
        EditText(this).apply {
          hint = if (sparkKey().isEmpty()) "Spark API key (optional)" else "Spark key saved · tap Save to replace"
          inputType = InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_VARIATION_PASSWORD
          setTextColor(Color.WHITE)
          setHintTextColor(MUTED)
          textSize = 13f
          layoutParams = LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f)
        }
    keyRow.addView(keyField)
    keyRow.addView(
        Button(this).apply {
          text = "Save"
          setOnClickListener {
            val v = keyField.text.toString().trim()
            getSharedPreferences("topo", MODE_PRIVATE).edit().putString("spark_key", v).apply()
            keyField.text.clear()
            keyField.hint =
                if (v.isEmpty()) "Spark API key (optional)" else "Spark key saved · tap Save to replace"
            setStatus(if (v.isEmpty()) "spark key cleared · offline grouping only"
                else "spark key saved · taps now ask spark")
          }
        })
    root.addView(keyRow)

    view =
        ImageView(this).apply {
          adjustViewBounds = true
          scaleType = ImageView.ScaleType.FIT_CENTER
          layoutParams =
              LinearLayout.LayoutParams(
                  ViewGroup.LayoutParams.MATCH_PARENT,
                  ViewGroup.LayoutParams.WRAP_CONTENT,
              )
          setOnTouchListener { v, event ->
            if (event.action == MotionEvent.ACTION_DOWN) {
              val drawable = (v as ImageView).drawable
              if (drawable != null) {
                val scale =
                    minOf(
                        v.width.toFloat() / drawable.intrinsicWidth,
                        v.height.toFloat() / drawable.intrinsicHeight,
                    )
                val padX = (v.width - drawable.intrinsicWidth * scale) / 2f
                val padY = (v.height - drawable.intrinsicHeight * scale) / 2f
                val x = (event.x - padX) / scale
                val y = (event.y - padY) / scale
                if (x >= 0 && y >= 0 && x < photo.width && y < photo.height) select(x, y)
              }
              v.performClick()
            }
            true
          }
        }
    root.addView(view)
    setContentView(root)
  }

  private fun setStatus(text: String) {
    Log.i(TAG, text)
    runOnUiThread { status.text = text }
  }

  private fun loadAsset(name: String): Bitmap =
      assets.open(name).use {
        BitmapFactory.decodeStream(
            it,
            null,
            BitmapFactory.Options().apply { inPreferredConfig = Bitmap.Config.ARGB_8888 },
        )
      } ?: error("cannot decode $name")

  private fun dp(value: Int): Int = (value * resources.displayMetrics.density).toInt()
}
