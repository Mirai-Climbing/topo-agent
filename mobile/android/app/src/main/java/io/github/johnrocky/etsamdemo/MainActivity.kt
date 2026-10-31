package io.github.johnrocky.etsamdemo

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Typeface
import android.media.ExifInterface
import android.os.Bundle
import android.util.Log
import android.view.MotionEvent
import android.view.ViewGroup
import android.widget.Button
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.FileProvider
import java.io.File
import kotlin.concurrent.thread
import kotlin.math.exp
import org.pytorch.executorch.EValue
import org.pytorch.executorch.Module
import org.pytorch.executorch.Tensor

private const val TAG = "ETSAMDemo"

private const val BACKGROUND = 0xFF101014.toInt()
private const val ACCENT = 0xFF6FE3B4.toInt()
private const val MUTED = 0xFF9AA0A6.toInt()

private const val SIDE = 640
private const val PROTO = 160
private const val COEFS = 32
private const val ANCHORS = 8400
private const val CONF_THRESH = 0.5f
private const val NMS_THRESH = 0.5f
private const val MAX_DETS = 100
/** Detection and overlay run on this; the phone screen cannot show more anyway. */
private const val MAX_SIDE = 1400

private data class Det(
    val x1: Float,
    val y1: Float,
    val x2: Float,
    val y2: Float,
    val conf: Float,
    val coefs: FloatArray,
)

/**
 * One-shot climbing-hold detector. A YOLO11n-seg model trained on the Kaggle hold dataset runs
 * once per photo; every hold mask is drawn immediately. Tapping a hold only selects it for
 * inspection (index + confidence in the status line) -- detection needs no tap at all.
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
  private var captureFile = File("")
  @Volatile private var busy = true

  private val cameraLauncher =
      registerForActivityResult(ActivityResultContracts.TakePicture()) { ok ->
        if (ok) onCapture()
        else setStatus("capture cancelled · %d holds on current photo".format(dets.size))
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

  /** Camera photos carry EXIF rotation; apply it before anything else. */
  private fun onCapture() {
    thread {
      runCatching {
        busy = true
        val raw = BitmapFactory.decodeFile(captureFile.absolutePath) ?: error("cannot decode capture")
        val exif = ExifInterface(captureFile.absolutePath)
        val orient = exif.getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL)
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
        detect(modelFile() ?: error("model gone"), "camera")
      }.onFailure {
        Log.e(TAG, "capture failed", it)
        setStatus("capture failed: ${it.message}")
        busy = false
      }
    }
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

  /** Tint every mask; the selected hold (if any) is drawn last and brighter. */
  private fun render() {
    val start = System.nanoTime()
    val w = photo.width
    val h = photo.height
    val base = IntArray(w * h)
    photo.getPixels(base, 0, w, 0, 0, w, h)
    val out = base.copyOf()
    val scale = SIDE / maxOf(w, h).toFloat()
    val padX = (SIDE - w * scale) / 2f
    val padY = (SIDE - h * scale) / 2f
    val order = dets.indices.sortedBy { if (it == selected) 1 else 0 }
    for (idx in order) {
      val d = dets[idx]
      val color = palette(idx)
      val bright = idx == selected
      val alpha = if (bright) 0.75f else 0.45f
      val cr = (color shr 16 and 0xFF) * alpha
      val cg = (color shr 8 and 0xFF) * alpha
      val cb = (color and 0xFF) * alpha
      // Sigmoid mask at proto resolution over the box region (plus a 1-cell margin),
      // then bilinear-upsampled to photo pixels: smooth edges instead of 160-grid blocks.
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
          for (k in 0 until COEFS) v += d.coefs[k] * protoAt(k, p)
          small[cy * mw + cx] = 1f / (1f + exp(-v))
        }
      }
      val bx0 = d.x1.toInt().coerceIn(0, w - 1)
      val by0 = d.y1.toInt().coerceIn(0, h - 1)
      val bx1 = d.x2.toInt().coerceIn(0, w)
      val by1 = d.y2.toInt().coerceIn(0, h)
      val bw = bx1 - bx0
      val bh = by1 - by0
      if (bw <= 0 || bh <= 0) continue
      // Binary mask at photo resolution over the box.
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
      for (y in 0 until bh) {
        for (x in 0 until bw) {
          if (!bin[y * bw + x]) continue
          val o = (by0 + y) * w + (bx0 + x)
          val c = base[o]
          out[o] =
              Color.rgb(
                  ((c shr 16 and 0xFF) * (1 - alpha) + cr).toInt(),
                  ((c shr 8 and 0xFF) * (1 - alpha) + cg).toInt(),
                  ((c and 0xFF) * (1 - alpha) + cb).toInt(),
              )
        }
      }
    }
    Log.i(TAG, "render %.0f ms for %d masks".format((System.nanoTime() - start) / 1e6, dets.size))
    overlayBmp = Bitmap.createBitmap(out, w, h, Bitmap.Config.ARGB_8888)
    showOverlay = true
    updateView()
  }

  private var protoCache: FloatArray = FloatArray(0)

  private fun protoAt(ch: Int, p: Int): Float = protoCache[ch * PROTO * PROTO + p]

  private fun palette(i: Int): Int {
    val colors =
        intArrayOf(
            0xFF4FC3F7.toInt(), 0xFFAED581.toInt(), 0xFFFFB74D.toInt(), 0xFFBA68C8.toInt(),
            0xFFFF8A65.toInt(), 0xFF7986CB.toInt(), 0xFF4DB6AC.toInt(), 0xFFF06292.toInt(),
        )
    return colors[i % colors.size]
  }

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
    if (best < 0) return
    selected = best
    val d = dets[best]
    render()
    setStatus("hold #%d · conf %.2f · %d holds total".format(best, d.conf, dets.size))
    Log.i(TAG, "tap ($x, $y) -> hold #$best conf ${d.conf}")
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
          text = "YOLO11n-seg · one pass, every hold · tap to inspect"
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
    root.addView(
        Button(this).apply {
          text = "Camera"
          setOnClickListener { openCamera() }
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
