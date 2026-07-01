package xyz.block.gosling.features.agent

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.os.IBinder
import android.util.Log
import androidx.core.app.NotificationCompat
import androidx.core.content.ContextCompat
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import xyz.block.gosling.R
import xyz.block.gosling.features.app.MainActivity
import java.io.File
import java.net.HttpURLConnection
import java.net.URL

class LlamaRuntimeService : Service() {
    companion object {
        const val ACTION_START = "xyz.block.gosling.action.START_LLAMA_RUNTIME"
        const val ACTION_STOP = "xyz.block.gosling.action.STOP_LLAMA_RUNTIME"
        const val NOTIFICATION_CHANNEL_ID = "llama_runtime_service"
        const val NOTIFICATION_ID = 4
        const val MODEL_PATH = "/data/local/tmp/gguf/gemma-4-E2B-it-Q8_0.gguf"
        const val LOCAL_API_BASE = "http://127.0.0.1:8080"

        private const val TAG = "LlamaRuntimeService"
        private const val TMP_LLAMA_ROOT = "/data/local/tmp/llama.cpp"
        private const val TMP_LLAMA_SERVER = "$TMP_LLAMA_ROOT/bin/llama-server"
        private val ASSET_LIBRARIES = listOf(
            "vendor.qti.hardware.dsp@1.0.so"
        )

        fun start(context: Context) {
            val intent = Intent(context, LlamaRuntimeService::class.java).apply {
                action = ACTION_START
            }
            ContextCompat.startForegroundService(context, intent)
        }

        fun stop(context: Context) {
            val intent = Intent(context, LlamaRuntimeService::class.java).apply {
                action = ACTION_STOP
            }
            context.startService(intent)
        }

        fun isApiReachable(): Boolean {
            var connection: HttpURLConnection? = null
            return try {
                connection = (URL("$LOCAL_API_BASE/v1/models").openConnection() as HttpURLConnection).apply {
                    connectTimeout = 1200
                    readTimeout = 1200
                    requestMethod = "GET"
                }
                connection.responseCode in 200..299
            } catch (e: Exception) {
                false
            } finally {
                connection?.disconnect()
            }
        }
    }

    private val job = Job()
    private val scope = CoroutineScope(Dispatchers.IO + job)
    private var process: Process? = null
    private lateinit var logFile: File

    override fun onCreate() {
        super.onCreate()
        logFile = File(filesDir, "llama-server.log")
        createNotificationChannel()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        when (intent?.action) {
            ACTION_STOP -> {
                stopRuntime()
                stopSelf()
                return START_NOT_STICKY
            }
            else -> {
                startForeground(NOTIFICATION_ID, createNotification("Starting Gemma on llama.cpp"))
                startRuntime()
            }
        }

        return START_STICKY
    }

    override fun onDestroy() {
        stopRuntime()
        job.cancel()
        super.onDestroy()
    }

    override fun onBind(intent: Intent?): IBinder? = null

    private fun startRuntime() {
        if (isApiReachable()) {
            updateNotification("Gemma is already reachable")
            return
        }

        val existingProcess = process
        if (existingProcess?.isAlive == true) {
            updateNotification("Gemma is still loading")
            return
        }

        scope.launch {
            val server = resolveServerExecutable()
            if (server == null) {
                updateNotification("llama-server binary not found")
                Log.e(TAG, "No packaged or staged llama-server binary found")
                return@launch
            }

            val model = File(MODEL_PATH)
            if (!model.exists()) {
                updateNotification("Model file missing on phone")
                Log.e(TAG, "Model file not found at $MODEL_PATH")
                return@launch
            }

            try {
                logFile.parentFile?.mkdirs()
                if (logFile.exists() && logFile.length() > 2_000_000L) {
                    logFile.delete()
                }

                val runtimeLibDir = prepareRuntimeLibraryDir()
                val nativeLibDir = applicationInfo.nativeLibraryDir
                val tmpLibDir = "$TMP_LLAMA_ROOT/lib"
                val libraryPath = listOf(runtimeLibDir.absolutePath, nativeLibDir, tmpLibDir)
                    .filter { it.isNotBlank() }
                    .joinToString(":")

                val command = listOf(
                    server.absolutePath,
                    "--jinja",
                    "--host", "127.0.0.1",
                    "--port", "8080",
                    "--no-mmap",
                    "-m", MODEL_PATH,
                    "--poll", "1000",
                    "-t", "6",
                    "--cpu-mask", "0xfc",
                    "--cpu-strict", "1",
                    "--ctx-size", "4096",
                    "--ubatch-size", "512",
                    "-fa", "on",
                    "-ngl", "99",
                    "--device", "HTP0",
                    "-np", "1"
                )

                val builder = ProcessBuilder(command)
                    .directory(server.parentFile ?: filesDir)
                    .redirectErrorStream(true)
                    .redirectOutput(ProcessBuilder.Redirect.appendTo(logFile))

                builder.environment()["LD_LIBRARY_PATH"] = libraryPath
                builder.environment()["ADSP_LIBRARY_PATH"] = libraryPath
                builder.environment()["GGML_HEXAGON_HOSTBUF"] = "0"

                Log.i(TAG, "Starting llama.cpp runtime: ${command.joinToString(" ")}")
                process = builder.start()
                updateNotification("Loading Gemma 4 E2B on HTP0")

                waitUntilReady()
            } catch (e: Exception) {
                updateNotification("Failed to start llama.cpp: ${e.message}")
                Log.e(TAG, "Failed to start llama.cpp", e)
            }
        }
    }

    private suspend fun waitUntilReady() {
        repeat(90) {
            if (isApiReachable()) {
                updateNotification("Gemma ready on phone localhost")
                return
            }

            val currentProcess = process
            if (currentProcess != null && !currentProcess.isAlive) {
                updateNotification("llama.cpp exited while loading")
                Log.e(TAG, "llama.cpp exited. See ${logFile.absolutePath}")
                return
            }

            delay(2000)
        }

        updateNotification("Still loading Gemma")
    }

    private fun stopRuntime() {
        process?.destroy()
        process = null
        updateNotification("llama.cpp stopped")
    }

    private fun resolveServerExecutable(): File? {
        val packaged = File(applicationInfo.nativeLibraryDir, "libllama-server-bin.so")
        if (packaged.exists() && packaged.canExecute()) {
            return packaged
        }

        val staged = File(TMP_LLAMA_SERVER)
        if (staged.exists() && staged.canExecute()) {
            return staged
        }

        return null
    }

    private fun prepareRuntimeLibraryDir(): File {
        val dir = File(filesDir, "llama-libs")
        dir.mkdirs()

        ASSET_LIBRARIES.forEach { name ->
            val target = File(dir, name)
            if (!target.exists() || target.length() == 0L) {
                assets.open("llama-libs/$name").use { input ->
                    target.outputStream().use { output ->
                        input.copyTo(output)
                    }
                }
            }
            target.setReadable(true, false)
        }

        return dir
    }

    private fun createNotificationChannel() {
        val channel = NotificationChannel(
            NOTIFICATION_CHANNEL_ID,
            "Local llama.cpp runtime",
            NotificationManager.IMPORTANCE_LOW
        ).apply {
            description = "Runs the phone-local Gemma llama.cpp server"
        }
        getSystemService(NotificationManager::class.java).createNotificationChannel(channel)
    }

    private fun updateNotification(message: String) {
        val manager = getSystemService(NotificationManager::class.java)
        manager.notify(NOTIFICATION_ID, createNotification(message))
    }

    private fun createNotification(message: String): Notification {
        val notificationIntent = Intent(this, MainActivity::class.java)
        val pendingIntent = PendingIntent.getActivity(
            this,
            0,
            notificationIntent,
            PendingIntent.FLAG_IMMUTABLE
        )

        return NotificationCompat.Builder(this, NOTIFICATION_CHANNEL_ID)
            .setContentTitle("Local Gemma runtime")
            .setContentText(message)
            .setSmallIcon(R.drawable.ic_launcher_foreground)
            .setContentIntent(pendingIntent)
            .setOngoing(true)
            .build()
    }
}
