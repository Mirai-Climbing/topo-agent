package io.github.johnrocky.etsamdemo

import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner

/** Offline checks of the Spark request builder and answer parser (no network). */
@RunWith(RobolectricTestRunner::class)
class SparkRouteTest {
  private fun row(
      id: Int,
      hsv: FloatArray = floatArrayOf(0f, 0.9f, 0.8f),
      x: Float = 0.1f * id,
      y: Float = 0.5f,
  ): SparkRoute.HoldRow =
      SparkRoute.HoldRow(
          id = id, x = x, y = y, r = 0.03f, hsv = hsv,
          lab = floatArrayOf(50f, 40f, 30f), colorName = SparkRoute.colorName(hsv),
          hist = FloatArray(HIST_H * HIST_S).also { it[id % it.size] = 1f })

  @Test
  fun colorNamesFollowTheServerTable() {
    assertEquals("red", SparkRoute.colorName(floatArrayOf(0.0f, 0.9f, 0.8f)))
    assertEquals("blue", SparkRoute.colorName(floatArrayOf(0.6f, 0.9f, 0.8f)))
    assertEquals("white", SparkRoute.colorName(floatArrayOf(0.0f, 0.05f, 0.9f)))
    assertEquals("black", SparkRoute.colorName(floatArrayOf(0.0f, 0.1f, 0.1f)))
    assertEquals("gray", SparkRoute.colorName(floatArrayOf(0.0f, 0.1f, 0.5f)))
  }

  @Test
  fun colorDistanceSeparatesHues() {
    val red = floatArrayOf(0.0f, 0.9f, 0.8f)
    val red2 = floatArrayOf(0.01f, 0.85f, 0.78f)
    val blue = floatArrayOf(0.6f, 0.9f, 0.8f)
    assertTrue(SparkRoute.colorDistance(red, red2) < SparkRoute.SIMILAR)
    assertTrue(SparkRoute.colorDistance(red, blue) > 0.5f)
  }

  @Test
  fun tableMirrorsTheServerShape() {
    val rows = listOf(
        row(0, x = 0.2f), row(1, x = 0.25f), row(2, floatArrayOf(0.6f, 0.9f, 0.8f), x = 0.8f))
    val table = SparkRoute.buildTable(rows, 0)
    val lines = table.split("\n")
    assertEquals("id | color | hue sat val | L a b | group | x y size | near | hist", lines[0])
    assertTrue(lines[1].startsWith("0* | red | "))
    assertTrue("2 | blue | " in lines[3])
    assertTrue(table.contains("Color groups: "))
    assertTrue(table.contains(
        "Hist lists the holds whose color distribution best matches each row"))
    assertTrue(table.endsWith(
        "The user tapped hold 0 (marked 0*). Return ONLY the holds of the tapped hold's route as JSON."))
  }

  @Test
  fun requestBodyMatchesTheChatCompletionsContract() {
    val body = JSONObject(SparkRoute.requestBody("table", byteArrayOf(1), byteArrayOf(2), byteArrayOf(3)))
    assertEquals(SparkRoute.MODEL, body.getString("model"))
    assertEquals(0.2, body.getDouble("temperature"), 1e-9)
    assertEquals("low", body.getString("reasoning_effort"))
    val messages = body.getJSONArray("messages")
    assertEquals(2, messages.length())
    assertEquals("system", messages.getJSONObject(0).getString("role"))
    assertTrue("climbing route grouping engine" in messages.getJSONObject(0).getString("content"))
    val content = messages.getJSONObject(1).getJSONArray("content")
    assertEquals(4, content.length()) // table, photo, annotated, close-up
    assertEquals("table", content.getJSONObject(0).getString("text"))
    for (i in 1..3) {
      val url = content.getJSONObject(i).getJSONObject("image_url").getString("url")
      assertTrue(url.startsWith("data:image/jpeg;base64,"))
    }
  }

  @Test
  fun overlayIsMutableForCanvasDrawing() {
    val bmp = SparkRoute.newOverlay(IntArray(16) { 0xFF000000.toInt() }, 4, 4)
    assertTrue(bmp.isMutable) // regressed once: immutable bitmap crashed drawTapFrame
    assertEquals(4, bmp.width)
  }

  @Test
  fun parseAnswerAcceptsValidRoutes() {
    val ans = SparkRoute.parseAnswer(
        "{\"hold_ids\": [2, 0], \"confidence\": 0.9}", setOf(0, 1, 2), 0)
    assertEquals("", ans.error)
    assertEquals(listOf(0, 2), ans.ids) // tap first
    assertEquals(0.9, ans.confidence, 1e-9)
  }

  @Test
  fun parseAnswerUnwrapsFencesAndRejectsBadShapes() {
    val fenced = SparkRoute.parseAnswer(
        "```json\n{\"hold_ids\": [1], \"confidence\": 0.5}\n```", setOf(0, 1), 0)
    assertEquals("", fenced.error)
    assertEquals("not valid JSON",
        SparkRoute.parseAnswer("nope", setOf(0), 0).error)
    assertEquals("missing hold_ids or confidence",
        SparkRoute.parseAnswer("{\"hold_ids\": [], \"confidence\": 0.5}", setOf(0), 0).error)
    assertEquals("unknown id 7",
        SparkRoute.parseAnswer("{\"hold_ids\": [7], \"confidence\": 0.5}", setOf(0), 0).error)
    assertEquals("hold 1 is assigned twice",
        SparkRoute.parseAnswer(
            "{\"hold_ids\": [1, 1], \"confidence\": 0.5}", setOf(0, 1), 0).error)
  }
}
