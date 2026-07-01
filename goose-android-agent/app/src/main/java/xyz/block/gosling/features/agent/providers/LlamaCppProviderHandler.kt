package xyz.block.gosling.features.agent.providers

import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import org.json.JSONArray
import org.json.JSONObject
import xyz.block.gosling.features.agent.Content
import xyz.block.gosling.features.agent.InternalToolCall
import xyz.block.gosling.features.agent.Message
import xyz.block.gosling.features.agent.SerializableToolDefinitions
import xyz.block.gosling.features.agent.Tool
import xyz.block.gosling.features.agent.ToolCall
import xyz.block.gosling.features.agent.ToolDefinition
import xyz.block.gosling.features.agent.ToolFunctionDefinition
import xyz.block.gosling.features.agent.ToolHandler
import xyz.block.gosling.features.agent.ToolParameter
import xyz.block.gosling.features.agent.ToolParametersObject
import java.lang.reflect.Method

class LlamaCppProviderHandler : LLMProviderHandler {
    private val jsonFormat = Json {
        encodeDefaults = true
        ignoreUnknownKeys = true
    }

    override fun createToolDefinitions(toolMethods: List<Method>): SerializableToolDefinitions {
        val toolDefinitions = toolMethods.mapNotNull { method ->
            val tool = method.getAnnotation(Tool::class.java) ?: return@mapNotNull null

            val toolParameters = ToolParametersObject(
                properties = tool.parameters.associate { param ->
                    param.name to ToolParameter(
                        type = param.type,
                        description = param.description
                    )
                },
                required = tool.parameters
                    .filter { it.required }
                    .map { it.name }
            )

            ToolDefinition(
                function = ToolFunctionDefinition(
                    name = tool.name,
                    description = tool.description,
                    parameters = toolParameters
                )
            )
        }

        return SerializableToolDefinitions.OpenAITools(toolDefinitions)
    }

    override fun createRequest(
        modelIdentifier: String,
        messages: List<Message>,
        tools: SerializableToolDefinitions,
        apiKey: String?
    ): String {
        val jsonMessages = JSONArray()
        messages.forEach { message ->
            jsonMessages.put(
                JSONObject()
                    .put("role", message.role)
                    .put("content", message.content.asPlainText())
            )
        }
        val toolDefinitions = when (tools) {
            is SerializableToolDefinitions.OpenAITools -> tools.definitions
            else -> emptyList()
        }

        val request = JSONObject()
            .put("model", modelIdentifier)
            .put("messages", jsonMessages)
            .put("temperature", 0.1)
            .put("max_tokens", 256)

        if (toolDefinitions.isNotEmpty()) {
            request
                .put("tools", JSONArray(jsonFormat.encodeToString(toolDefinitions)))
                .put("tool_choice", "auto")
        }

        return request.toString()
    }

    override fun getApiUrl(modelIdentifier: String, apiKey: String?): String {
        return "http://127.0.0.1:8080/v1/chat/completions"
    }

    override fun getHeaders(apiKey: String?): Map<String, String> {
        return emptyMap()
    }

    override fun parseResponse(
        response: JSONObject,
        requestDurationMs: Double
    ): Triple<String, List<InternalToolCall>?, Map<String, Double>> {
        val assistantMessage = response.getJSONArray("choices").getJSONObject(0)
            .getJSONObject("message")
        val content = assistantMessage.optString("content")
            .ifBlank { assistantMessage.optString("reasoning_content", "Ok") }
        val tools = assistantMessage.optJSONArray("tool_calls")?.let {
            List(it.length()) { i -> ToolHandler.fromJson(it.getJSONObject(i)) }
        }
        val usage = response.optJSONObject("usage")
        val annotation = mapOf(
            "duration" to requestDurationMs,
            "input_tokens" to (usage?.optInt("prompt_tokens") ?: 0).toDouble(),
            "output_tokens" to (usage?.optInt("completion_tokens") ?: 0).toDouble()
        )

        return Triple(content, tools, annotation)
    }

    private fun List<Content>?.asPlainText(): String {
        return this?.joinToString("\n") { item ->
            when (item) {
                is Content.Text -> item.text
                is Content.ImageUrl ->
                    "[Image attachment omitted: the current phone-local llama.cpp Gemma GGUF path is text-only.]"
            }
        } ?: ""
    }
}
