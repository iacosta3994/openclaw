import type { StreamFn } from "@mariozechner/pi-agent-core";
import type { SimpleStreamOptions } from "@mariozechner/pi-ai";
import { streamSimple } from "@mariozechner/pi-ai";
import type { OpenClawConfig } from "../../config/config.js";
import { log } from "./logger.js";

const OPENROUTER_APP_HEADERS: Record<string, string> = {
  "HTTP-Referer": "https://openclaw.ai",
  "X-Title": "OpenClaw",
};
const ANTHROPIC_CONTEXT_1M_BETA = "context-1m-2025-08-07";
const ANTHROPIC_1M_MODEL_PREFIXES = ["claude-opus-4", "claude-sonnet-4"] as const;
// NOTE: We only force `store=true` for *direct* OpenAI Responses.
// Codex responses (chatgpt.com/backend-api/codex/responses) require `store=false`.
const OPENAI_RESPONSES_APIS = new Set(["openai-responses"]);
const OPENAI_RESPONSES_PROVIDERS = new Set(["openai"]);

/**
 * Resolve provider-specific extra params from model config.
 * Used to pass through stream params like temperature/maxTokens.
 *
 * @internal Exported for testing only
 */
export function resolveExtraParams(params: {
  cfg: OpenClawConfig | undefined;
  provider: string;
  modelId: string;
}): Record<string, unknown> | undefined {
  const modelKey = `${params.provider}/${params.modelId}`;
  const modelConfig = params.cfg?.agents?.defaults?.models?.[modelKey];
  return modelConfig?.params ? { ...modelConfig.params } : undefined;
}

type CacheRetention = "none" | "short" | "long";
type CacheRetentionStreamOptions = Partial<SimpleStreamOptions> & {
  cacheRetention?: CacheRetention;
};

/**
 * Resolve cacheRetention from extraParams, supporting both new `cacheRetention`
 * and legacy `cacheControlTtl` values for backwards compatibility.
 *
 * Mapping: "5m" → "short", "1h" → "long"
 *
 * Only applies to Anthropic provider (OpenRouter uses openai-completions API
 * with hardcoded cache_control, not the cacheRetention stream option).
 *
 * Defaults to "short" for Anthropic provider when not explicitly configured.
 */
function resolveCacheRetention(
  extraParams: Record<string, unknown> | undefined,
  provider: string,
): CacheRetention | undefined {
  if (provider !== "anthropic") {
    return undefined;
  }

  // Prefer new cacheRetention if present
  const newVal = extraParams?.cacheRetention;
  if (newVal === "none" || newVal === "short" || newVal === "long") {
    return newVal;
  }

  // Fall back to legacy cacheControlTtl with mapping
  const legacy = extraParams?.cacheControlTtl;
  if (legacy === "5m") {
    return "short";
  }
  if (legacy === "1h") {
    return "long";
  }

  // Default to "short" for Anthropic when not explicitly configured
  return "short";
}

function createStreamFnWithExtraParams(
  baseStreamFn: StreamFn | undefined,
  extraParams: Record<string, unknown> | undefined,
  provider: string,
): StreamFn | undefined {
  if (!extraParams || Object.keys(extraParams).length === 0) {
    return undefined;
  }

  const streamParams: CacheRetentionStreamOptions = {};
  if (typeof extraParams.temperature === "number") {
    streamParams.temperature = extraParams.temperature;
  }
  if (typeof extraParams.maxTokens === "number") {
    streamParams.maxTokens = extraParams.maxTokens;
  }
  const cacheRetention = resolveCacheRetention(extraParams, provider);
  if (cacheRetention) {
    streamParams.cacheRetention = cacheRetention;
  }

  if (Object.keys(streamParams).length === 0) {
    return undefined;
  }

  log.debug(`creating streamFn wrapper with params: ${JSON.stringify(streamParams)}`);

  const underlying = baseStreamFn ?? streamSimple;
  const wrappedStreamFn: StreamFn = (model, context, options) =>
    underlying(model, context, {
      ...streamParams,
      ...options,
    });

  return wrappedStreamFn;
}

function isDirectOpenAIBaseUrl(baseUrl: unknown): boolean {
  if (typeof baseUrl !== "string" || !baseUrl.trim()) {
    return true;
  }

  try {
    const host = new URL(baseUrl).hostname.toLowerCase();
    return host === "api.openai.com" || host === "chatgpt.com";
  } catch {
    const normalized = baseUrl.toLowerCase();
    return normalized.includes("api.openai.com") || normalized.includes("chatgpt.com");
  }
}

function shouldForceResponsesStore(model: {
  api?: unknown;
  provider?: unknown;
  baseUrl?: unknown;
}): boolean {
  if (typeof model.api !== "string" || typeof model.provider !== "string") {
    return false;
  }
  if (!OPENAI_RESPONSES_APIS.has(model.api)) {
    return false;
  }
  if (!OPENAI_RESPONSES_PROVIDERS.has(model.provider)) {
    return false;
  }
  return isDirectOpenAIBaseUrl(model.baseUrl);
}

function createOpenAIResponsesStoreWrapper(baseStreamFn: StreamFn | undefined): StreamFn {
  const underlying = baseStreamFn ?? streamSimple;
  return (model, context, options) => {
    if (!shouldForceResponsesStore(model)) {
      return underlying(model, context, options);
    }

    const originalOnPayload = options?.onPayload;
    return underlying(model, context, {
      ...options,
      onPayload: (payload) => {
        if (payload && typeof payload === "object") {
          (payload as { store?: unknown }).store = true;
        }
        originalOnPayload?.(payload);
      },
    });
  };
}

function isAnthropic1MModel(modelId: string): boolean {
  const normalized = modelId.trim().toLowerCase();
  return ANTHROPIC_1M_MODEL_PREFIXES.some((prefix) => normalized.startsWith(prefix));
}

function parseHeaderList(value: unknown): string[] {
  if (typeof value !== "string") {
    return [];
  }
  return value
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);
}

function resolveAnthropicBetas(
  extraParams: Record<string, unknown> | undefined,
  provider: string,
  modelId: string,
): string[] | undefined {
  if (provider !== "anthropic") {
    return undefined;
  }

  const betas = new Set<string>();
  const configured = extraParams?.anthropicBeta;
  if (typeof configured === "string" && configured.trim()) {
    betas.add(configured.trim());
  } else if (Array.isArray(configured)) {
    for (const beta of configured) {
      if (typeof beta === "string" && beta.trim()) {
        betas.add(beta.trim());
      }
    }
  }

  if (extraParams?.context1m === true) {
    if (isAnthropic1MModel(modelId)) {
      betas.add(ANTHROPIC_CONTEXT_1M_BETA);
    } else {
      log.warn(`ignoring context1m for non-opus/sonnet model: ${provider}/${modelId}`);
    }
  }

  return betas.size > 0 ? [...betas] : undefined;
}

function mergeAnthropicBetaHeader(
  headers: Record<string, string> | undefined,
  betas: string[],
): Record<string, string> {
  const merged = { ...headers };
  const existingKey = Object.keys(merged).find((key) => key.toLowerCase() === "anthropic-beta");
  const existing = existingKey ? parseHeaderList(merged[existingKey]) : [];
  const values = Array.from(new Set([...existing, ...betas]));
  const key = existingKey ?? "anthropic-beta";
  merged[key] = values.join(",");
  return merged;
}

function createAnthropicBetaHeadersWrapper(
  baseStreamFn: StreamFn | undefined,
  betas: string[],
): StreamFn {
  const underlying = baseStreamFn ?? streamSimple;
  return (model, context, options) =>
    underlying(model, context, {
      ...options,
      headers: mergeAnthropicBetaHeader(options?.headers, betas),
    });
}

/**
 * Create a streamFn wrapper that forwards custom headers from provider config.
 * Allows users to set provider-level headers (e.g., `x-grok-conv-id` for xAI
 * cache grouping) in their model config and have them forwarded to the API.
 */
function createProviderHeadersWrapper(
  baseStreamFn: StreamFn | undefined,
  providerHeaders: Record<string, string>,
): StreamFn {
  const underlying = baseStreamFn ?? streamSimple;
  return (model, context, options) =>
    underlying(model, context, {
      ...options,
      headers: { ...providerHeaders, ...options?.headers },
    });
}

/**
 * Create a streamFn wrapper that adds OpenRouter app attribution headers.
 * These headers allow OpenClaw to appear on OpenRouter's leaderboard.
 */
function createOpenRouterHeadersWrapper(baseStreamFn: StreamFn | undefined): StreamFn {
  const underlying = baseStreamFn ?? streamSimple;
  return (model, context, options) =>
    underlying(model, context, {
      ...options,
      headers: {
        ...OPENROUTER_APP_HEADERS,
        ...options?.headers,
      },
    });
}

/**
 * Create a streamFn wrapper that injects tool_stream=true for Z.AI providers.
 *
 * Z.AI's API supports the `tool_stream` parameter to enable real-time streaming
 * of tool call arguments and reasoning content. When enabled, the API returns
 * progressive tool_call deltas, allowing users to see tool execution in real-time.
 *
 * @see https://docs.z.ai/api-reference#streaming
 */
function createZaiToolStreamWrapper(
  baseStreamFn: StreamFn | undefined,
  enabled: boolean,
): StreamFn {
  const underlying = baseStreamFn ?? streamSimple;
  return (model, context, options) => {
    if (!enabled) {
      return underlying(model, context, options);
    }

    const originalOnPayload = options?.onPayload;
    return underlying(model, context, {
      ...options,
      onPayload: (payload) => {
        if (payload && typeof payload === "object") {
          // Inject tool_stream: true for Z.AI API
          (payload as Record<string, unknown>).tool_stream = true;
        }
        originalOnPayload?.(payload);
      },
    });
  };
}

/**
 * Apply extra params (like temperature) to an agent's streamFn.
 * Also adds OpenRouter app attribution headers when using the OpenRouter provider.
 *
 * @internal Exported for testing
 */
export function applyExtraParamsToAgent(
  agent: { streamFn?: StreamFn },
  cfg: OpenClawConfig | undefined,
  provider: string,
  modelId: string,
  extraParamsOverride?: Record<string, unknown>,
  providerHeaders?: Record<string, string>,
  options?: { isReasoningModel?: boolean; previousResponseId?: string },
): void {
  const extraParams = resolveExtraParams({
    cfg,
    provider,
    modelId,
  });
  const override =
    extraParamsOverride && Object.keys(extraParamsOverride).length > 0
      ? Object.fromEntries(
          Object.entries(extraParamsOverride).filter(([, value]) => value !== undefined),
        )
      : undefined;
  const merged = Object.assign({}, extraParams, override);
  const wrappedStreamFn = createStreamFnWithExtraParams(agent.streamFn, merged, provider);

  if (wrappedStreamFn) {
    log.debug(`applying extraParams to agent streamFn for ${provider}/${modelId}`);
    agent.streamFn = wrappedStreamFn;
  }

  const anthropicBetas = resolveAnthropicBetas(merged, provider, modelId);
  if (anthropicBetas?.length) {
    log.debug(
      `applying Anthropic beta header for ${provider}/${modelId}: ${anthropicBetas.join(",")}`,
    );
    agent.streamFn = createAnthropicBetaHeadersWrapper(agent.streamFn, anthropicBetas);
  }

  if (provider === "openrouter") {
    log.debug(`applying OpenRouter app attribution headers for ${provider}/${modelId}`);
    agent.streamFn = createOpenRouterHeadersWrapper(agent.streamFn);
  }

  if (providerHeaders && Object.keys(providerHeaders).length > 0) {
    log.debug(`applying provider custom headers for ${provider}/${modelId}`);
    agent.streamFn = createProviderHeadersWrapper(agent.streamFn, providerHeaders);
  }

  // Enable Z.AI tool_stream for real-time tool call streaming.
  // Enabled by default for Z.AI provider, can be disabled via params.tool_stream: false
  if (provider === "zai" || provider === "z-ai") {
    const toolStreamEnabled = merged?.tool_stream !== false;
    if (toolStreamEnabled) {
      log.debug(`enabling Z.AI tool_stream for ${provider}/${modelId}`);
      agent.streamFn = createZaiToolStreamWrapper(agent.streamFn, true);
    }
  }

  // For xAI reasoning models, request encrypted reasoning content so it can be
  // replayed on subsequent turns to reduce re-reasoning costs.
  // DISABLED: pi-ai discards the encrypted content from the response stream,
  // so requesting it just increases response payload size with no benefit.
  // Re-enable when pi-ai is patched to expose encrypted reasoning content.
  // if (provider === "xai" && options?.isReasoningModel) {
  //   log.debug(`requesting encrypted reasoning content for ${provider}/${modelId}`);
  //   const prevStreamFn = agent.streamFn ?? streamSimple;
  //   agent.streamFn = (model, context, streamOpts) => {
  //     const origOnPayload = streamOpts?.onPayload;
  //     return prevStreamFn(model, context, {
  //       ...streamOpts,
  //       onPayload: (payload) => {
  //         if (payload && typeof payload === "object") {
  //           (payload as { include?: string[] }).include = ["reasoning.encrypted_content"];
  //         }
  //         origOnPayload?.(payload);
  //       },
  //     });
  //   };
  // }

  // Apply explicit store parameter when provided (e.g., store=false for cron/subagent sessions).
  // When no explicit store is set, default to store=true so xAI retains responses
  // server-side for previous_response_id chaining on the next turn.
  // (pi-ai hardcodes store=false for Responses API, which breaks chaining.)
  const explicitStore = extraParamsOverride?.store ?? true;
  if (typeof explicitStore === "boolean") {
    log.debug(`applying explicit store=${explicitStore} for ${provider}/${modelId}`);
    const prevStreamFn = agent.streamFn ?? streamSimple;
    agent.streamFn = (model, context, streamOpts) => {
      const origOnPayload = streamOpts?.onPayload;
      return prevStreamFn(model, context, {
        ...streamOpts,
        onPayload: (payload) => {
          if (payload && typeof payload === "object") {
            (payload as { store?: boolean }).store = explicitStore;
          }
          origOnPayload?.(payload);
        },
      });
    };
  }

  // Inject previous_response_id for Responses API conversation chaining.
  // On the FIRST API call of each run, this:
  // 1. Sets previous_response_id so the server uses its stored conversation state
  // 2. Moves the system/developer prompt from input[] to the `instructions` field
  //    (so the model always sees the latest system prompt, not the stale cached one)
  // 3. Trims input[] to only the last user message (the new prompt)
  // This reduces per-turn input tokens from ~full context to ~system prompt + new message.
  //
  // Subsequent calls within the same run (tool-use continuations) use full history
  // as normal, since we don't have the intermediate response IDs to chain them.
  const previousResponseId = options?.previousResponseId;
  if (typeof previousResponseId === "string" && previousResponseId.length > 0) {
    log.debug(`activating previous_response_id chaining for ${provider}/${modelId}`);
    const prevStreamFn = agent.streamFn ?? streamSimple;
    let callCount = 0;
    agent.streamFn = (model, context, streamOpts) => {
      callCount++;
      if (callCount > 1) {
        // Tool-use continuation — use full history without previous_response_id
        return prevStreamFn(model, context, streamOpts);
      }
      const origOnPayload = streamOpts?.onPayload;
      return prevStreamFn(model, context, {
        ...streamOpts,
        onPayload: (payload) => {
          if (payload && typeof payload === "object") {
            const p = payload as {
              previous_response_id?: string;
              instructions?: string;
              input?: Array<{ role?: string; type?: string; content?: unknown }>;
            };
            p.previous_response_id = previousResponseId;

            // Move the system/developer message to `instructions` so it gets
            // updated on each turn (previous_response_id reuses old instructions).
            const input = p.input;
            if (Array.isArray(input) && input.length > 0) {
              const firstItem = input[0];
              if (
                firstItem &&
                (firstItem.role === "system" || firstItem.role === "developer") &&
                typeof firstItem.content === "string"
              ) {
                p.instructions = firstItem.content;
              }
              // Keep only the last user message in input (the new prompt).
              // Everything else is already in the server's conversation state.
              const lastUserIndex = input.findLastIndex((item) => item.role === "user");
              if (lastUserIndex >= 0) {
                p.input = input.slice(lastUserIndex);
              }
            }

            log.debug(
              `previous_response_id injected: ${previousResponseId.slice(0, 20)}... ` +
                `input trimmed from ${input?.length ?? 0} to ${p.input?.length ?? 0} items`,
            );
          }
          origOnPayload?.(payload);
        },
      });
    };
  }

  // Work around upstream pi-ai hardcoding `store: false` for Responses API.
  // Force `store=true` for direct OpenAI/OpenAI Codex providers so multi-turn
  // server-side conversation state is preserved.
  // NOTE: This only activates for direct OpenAI providers (shouldForceResponsesStore),
  // so it won't override the explicit store=false set above for xAI/other providers.
  agent.streamFn = createOpenAIResponsesStoreWrapper(agent.streamFn);
}
