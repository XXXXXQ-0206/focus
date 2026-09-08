//! Production-friendly process adapter for any model endpoint or local model.

use std::{
    collections::{BTreeMap, VecDeque},
    io::Read,
    io::Write,
    path::PathBuf,
    process::{Command, Stdio},
    thread,
    time::{Duration, Instant},
};

use command_group::{CommandGroup, GroupChild};
use focus_kernel::{
    CancellationBridge, CancellationSignal, KernelError, Message, ModelEvent, ModelEventStream,
    ModelProvider, ModelRequest, ModelResponse, Role, ToolCall,
};
use reqwest::{
    Certificate, Client,
    header::{CONTENT_TYPE, HeaderName, HeaderValue},
};
use serde_json::{Value, json};
use uuid::Uuid;

use crate::cancellation::wait_for_cancellation;

const DEFAULT_HTTP_TIMEOUT: Duration = Duration::from_secs(120);
const DEFAULT_HTTP_RESPONSE_LIMIT: usize = 8 * 1024 * 1024;
const DEFAULT_COMMAND_TIMEOUT: Duration = Duration::from_secs(120);
const DEFAULT_COMMAND_OUTPUT_LIMIT: usize = 8 * 1024 * 1024;
const HTTP_RESPONSE_DIAGNOSTIC_BYTES: usize = 1_024;
const DEFAULT_OPENAI_MAX_RETRIES: u32 = 2;
const DEFAULT_OPENAI_RETRY_BASE_DELAY: Duration = Duration::from_millis(250);
const MAX_OPENAI_RETRY_DELAY: Duration = Duration::from_secs(5);

/// Wire contract used by an OpenAI-compatible endpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpenAiWireApi {
    /// The legacy `/chat/completions` request and `[DONE]` terminal marker.
    ChatCompletions,
    /// The `/responses` request and `response.completed` terminal marker.
    Responses,
}

/// Configuration for an OpenAI-compatible endpoint.
#[derive(Clone)]
pub struct OpenAiConfig {
    /// Base URL ending at the API version, such as `https://api.openai.com/v1`.
    pub base_url: String,
    /// Model identifier sent in every request.
    pub model: String,
    /// Provider wire contract. Chat Completions remains the compatibility default.
    pub wire_api: OpenAiWireApi,
    /// Optional bearer credential.
    pub api_key: Option<String>,
    /// Additional non-secret or provider-specific headers.
    pub extra_headers: BTreeMap<String, String>,
    /// End-to-end request timeout.
    pub timeout: Duration,
    /// Maximum accepted response body size.
    pub max_response_bytes: usize,
    /// Maximum number of transient request retries.
    pub max_retries: u32,
    /// Initial delay used for exponential retry backoff.
    pub retry_base_delay: Duration,
}

impl std::fmt::Debug for OpenAiConfig {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("OpenAiConfig")
            .field("base_url", &self.base_url)
            .field("model", &self.model)
            .field("wire_api", &self.wire_api)
            .field("api_key_configured", &self.api_key.is_some())
            .field("extra_header_names", &self.extra_headers.keys())
            .field("timeout", &self.timeout)
            .field("max_response_bytes", &self.max_response_bytes)
            .field("max_retries", &self.max_retries)
            .field("retry_base_delay", &self.retry_base_delay)
            .finish()
    }
}

impl OpenAiConfig {
    /// Create a provider configuration with bounded defaults.
    #[must_use]
    pub fn new(base_url: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into(),
            model: model.into(),
            wire_api: OpenAiWireApi::ChatCompletions,
            api_key: None,
            extra_headers: BTreeMap::new(),
            timeout: DEFAULT_HTTP_TIMEOUT,
            max_response_bytes: DEFAULT_HTTP_RESPONSE_LIMIT,
            max_retries: DEFAULT_OPENAI_MAX_RETRIES,
            retry_base_delay: DEFAULT_OPENAI_RETRY_BASE_DELAY,
        }
    }

    /// Attach a bearer credential.
    #[must_use]
    pub fn with_api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    /// Select the provider wire contract.
    #[must_use]
    pub fn with_wire_api(mut self, wire_api: OpenAiWireApi) -> Self {
        self.wire_api = wire_api;
        self
    }

    /// Replace the request timeout.
    #[must_use]
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Add a provider-specific request header.
    #[must_use]
    pub fn with_header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.extra_headers.insert(name.into(), value.into());
        self
    }

    /// Set the bounded transient retry count.
    #[must_use]
    pub fn with_max_retries(mut self, retries: u32) -> Self {
        self.max_retries = retries;
        self
    }

    /// Set the initial transient retry backoff.
    #[must_use]
    pub fn with_retry_base_delay(mut self, delay: Duration) -> Self {
        self.retry_base_delay = delay;
        self
    }
}

/// Direct OpenAI-compatible provider using the Runtime's normalized message contract.
#[derive(Clone)]
pub struct OpenAiCompatibleProvider {
    config: OpenAiConfig,
    client: Client,
    endpoint: reqwest::Url,
}

impl std::fmt::Debug for OpenAiCompatibleProvider {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("OpenAiCompatibleProvider")
            .field("config", &self.config)
            .field("endpoint", &self.endpoint)
            .finish_non_exhaustive()
    }
}

fn bundled_webpki_root_certificates() -> Result<Vec<Certificate>, KernelError> {
    webpki_root_certs::TLS_SERVER_ROOT_CERTS
        .iter()
        .map(|certificate| {
            Certificate::from_der(certificate.as_ref()).map_err(|error| {
                KernelError::Model(format!("bundled TLS root certificate is invalid: {error}"))
            })
        })
        .collect()
}

impl OpenAiCompatibleProvider {
    /// Validate configuration and build the reusable HTTP client.
    pub fn new(config: OpenAiConfig) -> Result<Self, KernelError> {
        if config.model.trim().is_empty() {
            return Err(KernelError::Model("OpenAI model must not be empty".into()));
        }
        if config.timeout.is_zero() {
            return Err(KernelError::Model(
                "OpenAI request timeout must be positive".into(),
            ));
        }
        if config.max_response_bytes == 0 {
            return Err(KernelError::Model(
                "OpenAI response limit must be positive".into(),
            ));
        }
        let endpoint = openai_endpoint(&config.base_url, config.wire_api)?;
        for (name, value) in &config.extra_headers {
            HeaderName::from_bytes(name.as_bytes()).map_err(|error| {
                KernelError::Model(format!("invalid OpenAI header name `{name}`: {error}"))
            })?;
            HeaderValue::from_str(value).map_err(|error| {
                KernelError::Model(format!("invalid OpenAI header value for `{name}`: {error}"))
            })?;
        }
        let client = Client::builder()
            .timeout(config.timeout)
            .tls_certs_only(bundled_webpki_root_certificates()?)
            .build()
            .map_err(|error| KernelError::Model(format!("OpenAI client build failed: {error}")))?;
        Ok(Self {
            config,
            client,
            endpoint,
        })
    }

    fn post_json(&self, body: &Value) -> reqwest::RequestBuilder {
        let mut builder = self.client.post(self.endpoint.clone()).json(body);
        if let Some(api_key) = self.config.api_key.as_deref().filter(|key| !key.is_empty()) {
            builder = builder.bearer_auth(api_key);
        }
        for (name, value) in &self.config.extra_headers {
            builder = builder.header(name, value);
        }
        builder
    }
}

#[async_trait::async_trait]
impl ModelProvider for OpenAiCompatibleProvider {
    async fn stream(
        &self,
        request: ModelRequest,
        cancellation: &dyn CancellationSignal,
    ) -> Result<ModelEventStream, KernelError> {
        if cancellation.is_cancelled() {
            return Ok(model_event_stream(vec![ModelEvent::Cancelled]));
        }
        if self.config.wire_api == OpenAiWireApi::Responses {
            return self.stream_responses(request, cancellation).await;
        }
        let cancellation = CancellationBridge::from_signal(cancellation);
        let body = json!({
            "model": self.config.model,
            "stream": true,
            "messages": openai_messages(&request.messages),
            "tools": request.tools.iter().map(|tool| json!({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.input_schema,
                }
            })).collect::<Vec<_>>()
        });
        let state = ChatCompletionsStreamState {
            provider: self.clone(),
            body,
            request_id: Uuid::new_v4().to_string(),
            turn_state: None,
            cancellation,
            attempt: 0,
            retry_delay: None,
            pending: VecDeque::new(),
            response: None,
            decoder: None,
            finished: false,
        };
        Ok(Box::pin(futures::stream::unfold(
            state,
            |mut state| async move {
                loop {
                    if let Some(event) = state.pending.pop_front() {
                        return Some((Ok(event), state));
                    }
                    if let Some(event) = state
                        .decoder
                        .as_mut()
                        .and_then(|decoder| decoder.pending.pop_front())
                    {
                        let retryable_failure = state
                            .decoder
                            .as_ref()
                            .is_some_and(OpenAiSseDecoder::can_retry_without_duplicate_output);
                        match &event {
                            ModelEvent::Completed { response }
                                if empty_assistant_completion(response) =>
                            {
                                state.retry_or_fail(
                                    KernelError::Model(
                                        "OpenAI stream completed without assistant text or tool calls"
                                            .into(),
                                    ),
                                    true,
                                );
                                continue;
                            }
                            ModelEvent::Failed { error } if retryable_failure => {
                                state.retry_or_fail(KernelError::Model(error.clone()), true);
                                continue;
                            }
                            _ => return Some((Ok(event), state)),
                        }
                    }
                    if state.finished {
                        return None;
                    }
                    if let Some(delay) = state.retry_delay.take()
                        && wait_retry(&state.cancellation, delay).await
                    {
                        state.pending.push_back(ModelEvent::Cancelled);
                        state.finished = true;
                        continue;
                    }
                    if state.cancellation.is_cancelled() {
                        state.pending.push_back(ModelEvent::Cancelled);
                        state.finished = true;
                        continue;
                    }
                    if state.response.is_none() {
                        let mut builder = state.provider.post_json(&state.body);
                        builder = builder.header("x-client-request-id", &state.request_id);
                        if let Some(turn_state) = state.turn_state.as_deref() {
                            builder = builder.header("x-codex-turn-state", turn_state);
                        }
                        let sent = tokio::select! {
                            () = wait_for_cancellation(&state.cancellation) => None,
                            result = builder.send() => Some(result),
                        };
                        let Some(sent) = sent else {
                            state.pending.push_back(ModelEvent::Cancelled);
                            state.finished = true;
                            continue;
                        };
                        let mut response = match sent {
                            Ok(response) => response,
                            Err(error) => {
                                state.retry_or_fail(
                                    KernelError::Model(format!(
                                        "OpenAI request failed: {}",
                                        describe_http_error(&error)
                                    )),
                                    true,
                                );
                                continue;
                            }
                        };
                        let status = response.status();
                        let content_type = response
                            .headers()
                            .get(CONTENT_TYPE)
                            .and_then(|value| value.to_str().ok())
                            .unwrap_or("<missing>")
                            .to_owned();
                        if !status.is_success() {
                            let error = match read_response_bytes(
                                &mut response,
                                &state.cancellation,
                                state.provider.config.max_response_bytes,
                            )
                            .await
                            {
                                Ok(bytes) => openai_response_error(
                                    status.as_u16(),
                                    &content_type,
                                    &bytes,
                                    "HTTP status was not successful",
                                    state.provider.config.api_key.as_deref(),
                                ),
                                Err(KernelError::Cancelled) => {
                                    state.pending.push_back(ModelEvent::Cancelled);
                                    state.finished = true;
                                    continue;
                                }
                                Err(error) => error,
                            };
                            if retryable_status(status)
                                && state.attempt < state.provider.config.max_retries
                            {
                                let delay = retry_delay(
                                    state.provider.config.retry_base_delay,
                                    state.attempt,
                                );
                                state.attempt = state.attempt.saturating_add(1);
                                state.pending.push_back(ModelEvent::RetryScheduled {
                                    attempt: state.attempt,
                                    delay_ms: duration_millis(delay),
                                    reason: format!("HTTP status {}", status.as_u16()),
                                });
                                state.retry_delay = Some(delay);
                            } else {
                                state.retry_or_fail(error, false);
                            }
                            continue;
                        }
                        if content_type
                            .split(';')
                            .next()
                            .is_some_and(|value| value.eq_ignore_ascii_case("text/event-stream"))
                        {
                            if state.turn_state.is_none() {
                                state.turn_state = response
                                    .headers()
                                    .get("x-codex-turn-state")
                                    .and_then(|value| value.to_str().ok())
                                    .filter(|value| !value.is_empty())
                                    .map(str::to_owned);
                            }
                            state.decoder = Some(OpenAiSseDecoder::new(
                                "openai-compatible".into(),
                                state.provider.config.model.clone(),
                                state.provider.endpoint.to_string(),
                                state.provider.config.api_key.clone(),
                                state.provider.config.max_response_bytes,
                            ));
                            state.response = Some(response);
                            continue;
                        }
                        let result = match read_response_bytes(
                            &mut response,
                            &state.cancellation,
                            state.provider.config.max_response_bytes,
                        )
                        .await
                        {
                            Ok(bytes) => match parse_openai_response(&bytes) {
                                Ok(response) => response,
                                Err(error) => {
                                    state.retry_or_fail(
                                        openai_response_error(
                                            status.as_u16(),
                                            &content_type,
                                            &bytes,
                                            &error.to_string(),
                                            state.provider.config.api_key.as_deref(),
                                        ),
                                        false,
                                    );
                                    continue;
                                }
                            },
                            Err(KernelError::Cancelled) => {
                                state.pending.push_back(ModelEvent::Cancelled);
                                state.finished = true;
                                continue;
                            }
                            Err(error) => {
                                state.retry_or_fail(error, true);
                                continue;
                            }
                        };
                        if empty_assistant_completion(&result) {
                            state.retry_or_fail(
                                KernelError::Model(
                                    "OpenAI response completed without assistant text or tool calls"
                                        .into(),
                                ),
                                true,
                            );
                        } else {
                            state.pending.extend(completed_response_events(
                                "openai-compatible",
                                &state.provider.config.model,
                                state.provider.endpoint.as_ref(),
                                result,
                            ));
                            state.finished = true;
                        }
                        continue;
                    }
                    if state
                        .decoder
                        .as_ref()
                        .is_some_and(|decoder| decoder.terminal)
                    {
                        state.response = None;
                        state.finished = true;
                        continue;
                    }
                    let next = {
                        let response = state.response.as_mut().expect("response must be active");
                        tokio::select! {
                            () = wait_for_cancellation(&state.cancellation) => None,
                            chunk = response.chunk() => Some(chunk),
                        }
                    };
                    match next {
                        Some(Ok(Some(chunk))) => state
                            .decoder
                            .as_mut()
                            .expect("decoder must exist with an active response")
                            .push_chunk(&chunk),
                        Some(Ok(None)) => state
                            .decoder
                            .as_mut()
                            .expect("decoder must exist with an active response")
                            .finish_eof(),
                        Some(Err(error)) => {
                            let error =
                                KernelError::Model(format!("OpenAI response read failed: {error}"));
                            if state
                                .decoder
                                .as_ref()
                                .is_some_and(OpenAiSseDecoder::can_retry_without_duplicate_output)
                            {
                                state.retry_or_fail(error, true);
                            } else {
                                state
                                    .decoder
                                    .as_mut()
                                    .expect("decoder must exist with an active response")
                                    .fail(error);
                            }
                        }
                        None => {
                            state
                                .decoder
                                .as_mut()
                                .expect("decoder must exist with an active response")
                                .cancel();
                        }
                    }
                }
            },
        )))
    }
}

struct ChatCompletionsStreamState {
    provider: OpenAiCompatibleProvider,
    body: Value,
    request_id: String,
    turn_state: Option<String>,
    cancellation: CancellationBridge,
    attempt: u32,
    retry_delay: Option<Duration>,
    pending: VecDeque<ModelEvent>,
    response: Option<reqwest::Response>,
    decoder: Option<OpenAiSseDecoder>,
    finished: bool,
}

impl ChatCompletionsStreamState {
    fn retry_or_fail(&mut self, error: KernelError, retryable: bool) {
        self.response = None;
        self.decoder = None;
        if retryable && self.attempt < self.provider.config.max_retries {
            let delay = retry_delay(self.provider.config.retry_base_delay, self.attempt);
            self.attempt = self.attempt.saturating_add(1);
            self.pending.push_back(ModelEvent::RetryScheduled {
                attempt: self.attempt,
                delay_ms: duration_millis(delay),
                reason: redact_secret(error.to_string(), self.provider.config.api_key.as_deref()),
            });
            self.retry_delay = Some(delay);
            return;
        }
        self.pending.push_back(ModelEvent::Failed {
            error: redact_secret(error.to_string(), self.provider.config.api_key.as_deref()),
        });
        self.finished = true;
    }
}

impl OpenAiCompatibleProvider {
    async fn stream_responses(
        &self,
        request: ModelRequest,
        cancellation: &dyn CancellationSignal,
    ) -> Result<ModelEventStream, KernelError> {
        let cancellation = CancellationBridge::from_signal(cancellation);
        let state = ResponsesStreamState {
            provider: self.clone(),
            body: openai_responses_request(&request, &self.config.model),
            request_id: Uuid::new_v4().to_string(),
            turn_state: None,
            cancellation,
            attempt: 0,
            retry_delay: None,
            pending: VecDeque::new(),
            response: None,
            decoder: None,
            finished: false,
        };
        Ok(Box::pin(futures::stream::unfold(
            state,
            |mut state| async move {
                loop {
                    if let Some(event) = state.pending.pop_front() {
                        return Some((Ok(event), state));
                    }
                    if let Some(event) = state
                        .decoder
                        .as_mut()
                        .and_then(|decoder| decoder.pending.pop_front())
                    {
                        return Some((Ok(event), state));
                    }
                    if state.finished {
                        return None;
                    }
                    if let Some(delay) = state.retry_delay.take()
                        && wait_retry(&state.cancellation, delay).await
                    {
                        state.pending.push_back(ModelEvent::Cancelled);
                        state.finished = true;
                        continue;
                    }
                    if state.cancellation.is_cancelled() {
                        state.pending.push_back(ModelEvent::Cancelled);
                        state.finished = true;
                        continue;
                    }
                    if state.response.is_none() {
                        let mut builder = state.provider.post_json(&state.body);
                        builder = builder.header("x-client-request-id", &state.request_id);
                        if let Some(turn_state) = state.turn_state.as_deref() {
                            builder = builder.header("x-codex-turn-state", turn_state);
                        }
                        let sent = tokio::select! {
                            () = wait_for_cancellation(&state.cancellation) => None,
                            result = builder.send() => Some(result),
                        };
                        let Some(sent) = sent else {
                            state.pending.push_back(ModelEvent::Cancelled);
                            state.finished = true;
                            continue;
                        };
                        let mut response = match sent {
                            Ok(response) => response,
                            Err(error) => {
                                state.retry_or_fail(
                                    KernelError::Model(format!(
                                        "OpenAI Responses request failed: {}",
                                        describe_http_error(&error)
                                    )),
                                    true,
                                );
                                continue;
                            }
                        };
                        let status = response.status();
                        let content_type = response
                            .headers()
                            .get(CONTENT_TYPE)
                            .and_then(|value| value.to_str().ok())
                            .unwrap_or("<missing>")
                            .to_owned();
                        if !status.is_success() {
                            let error = match read_response_bytes(
                                &mut response,
                                &state.cancellation,
                                state.provider.config.max_response_bytes,
                            )
                            .await
                            {
                                Ok(bytes) => openai_response_error(
                                    status.as_u16(),
                                    &content_type,
                                    &bytes,
                                    "Responses HTTP status was not successful",
                                    state.provider.config.api_key.as_deref(),
                                ),
                                Err(KernelError::Cancelled) => {
                                    state.pending.push_back(ModelEvent::Cancelled);
                                    state.finished = true;
                                    continue;
                                }
                                Err(error) => error,
                            };
                            state.retry_or_fail(error, retryable_status(status));
                            continue;
                        }
                        if !content_type
                            .split(';')
                            .next()
                            .is_some_and(|value| value.eq_ignore_ascii_case("text/event-stream"))
                        {
                            state.retry_or_fail(
                                KernelError::Model(format!(
                                    "OpenAI Responses stream expected text/event-stream, received {content_type}"
                                )),
                                false,
                            );
                            continue;
                        }
                        if state.turn_state.is_none() {
                            state.turn_state = response
                                .headers()
                                .get("x-codex-turn-state")
                                .and_then(|value| value.to_str().ok())
                                .filter(|value| !value.is_empty())
                                .map(str::to_owned);
                        }
                        state.decoder = Some(ResponsesSseDecoder::new(
                            "openai-compatible-responses".into(),
                            state.provider.config.model.clone(),
                            state.provider.endpoint.to_string(),
                            state.provider.config.max_response_bytes,
                        ));
                        state.response = Some(response);
                        continue;
                    }

                    let next = {
                        let response = state.response.as_mut().expect("response must be active");
                        tokio::select! {
                            () = wait_for_cancellation(&state.cancellation) => None,
                            chunk = response.chunk() => Some(chunk),
                        }
                    };
                    let Some(next) = next else {
                        state.pending.push_back(ModelEvent::Cancelled);
                        state.finished = true;
                        continue;
                    };
                    match next {
                        Ok(Some(chunk)) => {
                            let outcome = state
                                .decoder
                                .as_mut()
                                .expect("decoder must exist with an active response")
                                .push_chunk(&chunk);
                            match outcome {
                                Ok(()) => {
                                    if state
                                        .decoder
                                        .as_ref()
                                        .is_some_and(ResponsesSseDecoder::is_completed)
                                    {
                                        state.response = None;
                                        state.finished = true;
                                    }
                                }
                                Err(error) => state.retry_or_fail(error, false),
                            }
                        }
                        Ok(None) => {
                            if state
                                .decoder
                                .as_ref()
                                .is_some_and(ResponsesSseDecoder::is_completed)
                            {
                                state.response = None;
                                state.finished = true;
                            } else {
                                state.retry_or_fail(
                                    KernelError::Model(
                                        "OpenAI Responses stream ended before response.completed"
                                            .into(),
                                    ),
                                    true,
                                );
                            }
                        }
                        Err(error) => state.retry_or_fail(
                            KernelError::Model(format!("OpenAI response read failed: {error}")),
                            true,
                        ),
                    }
                }
            },
        )))
    }
}

struct ResponsesStreamState {
    provider: OpenAiCompatibleProvider,
    body: Value,
    request_id: String,
    turn_state: Option<String>,
    cancellation: CancellationBridge,
    attempt: u32,
    retry_delay: Option<Duration>,
    pending: VecDeque<ModelEvent>,
    response: Option<reqwest::Response>,
    decoder: Option<ResponsesSseDecoder>,
    finished: bool,
}

impl ResponsesStreamState {
    fn retry_or_fail(&mut self, error: KernelError, retryable: bool) {
        if let Some(decoder) = self.decoder.as_mut() {
            self.pending.append(&mut decoder.pending);
        }
        self.response = None;
        self.decoder = None;
        if retryable && self.attempt < self.provider.config.max_retries {
            let delay = retry_delay(self.provider.config.retry_base_delay, self.attempt);
            self.attempt = self.attempt.saturating_add(1);
            self.pending.push_back(ModelEvent::RetryScheduled {
                attempt: self.attempt,
                delay_ms: duration_millis(delay),
                reason: redact_secret(error.to_string(), self.provider.config.api_key.as_deref()),
            });
            self.retry_delay = Some(delay);
            return;
        }
        self.pending.push_back(ModelEvent::Failed {
            error: redact_secret(error.to_string(), self.provider.config.api_key.as_deref()),
        });
        self.finished = true;
    }
}

async fn read_response_bytes(
    response: &mut reqwest::Response,
    cancellation: &dyn CancellationSignal,
    max_response_bytes: usize,
) -> Result<Vec<u8>, KernelError> {
    let mut bytes = Vec::new();
    loop {
        let chunk = tokio::select! {
            () = wait_for_cancellation(cancellation) => {
                return Err(KernelError::Cancelled);
            }
            chunk = response.chunk() => chunk.map_err(|error| {
                KernelError::Model(format!("OpenAI response read failed: {error}"))
            })?,
        };
        let Some(chunk) = chunk else {
            return Ok(bytes);
        };
        bytes.extend_from_slice(&chunk);
        if bytes.len() > max_response_bytes {
            return Err(KernelError::Model(format!(
                "OpenAI response exceeded the {max_response_bytes} byte limit"
            )));
        }
    }
}

fn model_event_stream(events: Vec<ModelEvent>) -> ModelEventStream {
    Box::pin(futures::stream::iter(events.into_iter().map(Ok)))
}

fn retryable_status(status: reqwest::StatusCode) -> bool {
    matches!(status.as_u16(), 408 | 409 | 425 | 429) || status.is_server_error()
}

fn describe_http_error(error: &reqwest::Error) -> String {
    let category = if error.is_timeout() {
        "timeout"
    } else if error.is_connect() {
        "connect"
    } else if error.is_request() {
        "request"
    } else if error.is_builder() {
        "builder"
    } else {
        "other"
    };
    let mut description = format!("transport category={category}: {error}");
    let mut source = std::error::Error::source(error);
    while let Some(cause) = source {
        let detail = cause.to_string();
        if !detail.is_empty() && !description.contains(&detail) {
            description.push_str("; caused by: ");
            description.push_str(&detail);
        }
        source = std::error::Error::source(cause);
    }
    description
}

fn retry_delay(base: Duration, attempt: u32) -> Duration {
    let factor = 1_u32.checked_shl(attempt.min(5)).unwrap_or(32);
    base.saturating_mul(factor).min(MAX_OPENAI_RETRY_DELAY)
}

fn duration_millis(duration: Duration) -> u64 {
    duration.as_millis().min(u64::MAX as u128) as u64
}

async fn wait_retry(cancellation: &CancellationBridge, delay: Duration) -> bool {
    tokio::select! {
        () = wait_for_cancellation(cancellation) => true,
        () = tokio::time::sleep(delay) => false,
    }
}

fn empty_assistant_completion(response: &ModelResponse) -> bool {
    response.content.trim().is_empty() && response.tool_calls.is_empty()
}

fn completed_response_events(
    provider: &str,
    model: &str,
    endpoint: &str,
    response: ModelResponse,
) -> Vec<ModelEvent> {
    let mut events = vec![ModelEvent::RequestStarted {
        provider: provider.into(),
        model: model.into(),
        endpoint: endpoint.into(),
    }];
    if !response.content.is_empty() {
        events.push(ModelEvent::TextDelta {
            text: response.content.clone(),
        });
    }
    for call in &response.tool_calls {
        events.push(ModelEvent::ToolCallStarted {
            id: call.id.clone(),
            name: call.name.clone(),
        });
        events.push(ModelEvent::ToolCallReady { call: call.clone() });
    }
    events.push(ModelEvent::Completed { response });
    events
}

#[derive(Default)]
struct ResponsesToolCallBuffer {
    id: String,
    name: String,
    arguments: String,
    started: bool,
}

struct ResponsesSseDecoder {
    max_response_bytes: usize,
    received_bytes: usize,
    buffer: Vec<u8>,
    pending: VecDeque<ModelEvent>,
    content: String,
    tools: BTreeMap<String, ResponsesToolCallBuffer>,
    completed: bool,
}

impl ResponsesSseDecoder {
    fn new(provider: String, model: String, endpoint: String, max_response_bytes: usize) -> Self {
        Self {
            max_response_bytes,
            received_bytes: 0,
            buffer: Vec::new(),
            pending: VecDeque::from([ModelEvent::RequestStarted {
                provider,
                model,
                endpoint,
            }]),
            content: String::new(),
            tools: BTreeMap::new(),
            completed: false,
        }
    }

    fn is_completed(&self) -> bool {
        self.completed
    }

    fn push_chunk(&mut self, chunk: &[u8]) -> Result<(), KernelError> {
        if self.completed {
            return Ok(());
        }
        self.received_bytes = self.received_bytes.saturating_add(chunk.len());
        if self.received_bytes > self.max_response_bytes {
            return Err(KernelError::Model(format!(
                "OpenAI Responses stream exceeded the {} byte limit",
                self.max_response_bytes
            )));
        }
        self.buffer.extend_from_slice(chunk);
        while let Some((frame_end, delimiter_len)) = sse_frame_boundary(&self.buffer) {
            let frame = self.buffer.drain(..frame_end).collect::<Vec<_>>();
            self.buffer.drain(..delimiter_len);
            self.consume_frame(&frame)?;
            if self.completed {
                return Ok(());
            }
        }
        Ok(())
    }

    fn consume_frame(&mut self, frame: &[u8]) -> Result<(), KernelError> {
        let frame = std::str::from_utf8(frame).map_err(|error| {
            KernelError::Model(format!("OpenAI Responses SSE frame was not UTF-8: {error}"))
        })?;
        let data = frame
            .lines()
            .filter_map(|line| line.strip_prefix("data:"))
            .map(|line| line.strip_prefix(' ').unwrap_or(line))
            .collect::<Vec<_>>();
        if data.is_empty() {
            return Ok(());
        }
        let data = data.join("\n");
        if data == "[DONE]" {
            return Err(KernelError::Model(
                "OpenAI Responses stream ended without response.completed".into(),
            ));
        }
        let event: Value = serde_json::from_str(&data).map_err(|error| {
            KernelError::Model(format!("invalid OpenAI Responses SSE JSON: {error}"))
        })?;
        let kind = event
            .get("type")
            .and_then(Value::as_str)
            .ok_or_else(|| KernelError::Model("OpenAI Responses SSE event omitted type".into()))?;
        match kind {
            "response.output_text.delta" => {
                if let Some(delta) = event.get("delta").and_then(Value::as_str) {
                    self.content.push_str(delta);
                    self.pending
                        .push_back(ModelEvent::TextDelta { text: delta.into() });
                }
            }
            "response.reasoning_summary_text.delta" => {
                if let Some(delta) = event.get("delta").and_then(Value::as_str) {
                    self.pending
                        .push_back(ModelEvent::ReasoningSummaryDelta { text: delta.into() });
                }
            }
            "response.output_item.added" | "response.output_item.done" => {
                if let Some(item) = event.get("item") {
                    self.consume_tool_item(item)?;
                }
            }
            "response.function_call_arguments.delta" => {
                self.consume_tool_arguments_delta(&event)?;
            }
            "response.completed" => self.complete(&event)?,
            "response.failed" | "response.incomplete" => {
                let error = event
                    .get("response")
                    .and_then(|response| response.get("error"))
                    .or_else(|| event.get("error"))
                    .cloned()
                    .unwrap_or_else(|| Value::String(kind.into()));
                return Err(KernelError::Model(format!(
                    "OpenAI Responses stream reported {kind}: {error}"
                )));
            }
            _ => {}
        }
        Ok(())
    }

    fn consume_tool_item(&mut self, item: &Value) -> Result<(), KernelError> {
        if item.get("type").and_then(Value::as_str) != Some("function_call") {
            return Ok(());
        }
        let id = item
            .get("call_id")
            .or_else(|| item.get("id"))
            .and_then(Value::as_str)
            .filter(|value| !value.is_empty())
            .ok_or_else(|| {
                KernelError::Model("OpenAI Responses function call omitted call_id".into())
            })?;
        let call = self
            .tools
            .entry(id.into())
            .or_insert_with(|| ResponsesToolCallBuffer {
                id: id.into(),
                name: String::new(),
                arguments: String::new(),
                started: false,
            });
        if let Some(name) = item
            .get("name")
            .and_then(Value::as_str)
            .filter(|name| !name.is_empty())
        {
            call.name = name.into();
        }
        if let Some(arguments) = item.get("arguments").and_then(Value::as_str) {
            call.arguments = arguments.into();
        }
        if !call.started && !call.name.is_empty() {
            call.started = true;
            self.pending.push_back(ModelEvent::ToolCallStarted {
                id: call.id.clone(),
                name: call.name.clone(),
            });
        }
        Ok(())
    }

    fn consume_tool_arguments_delta(&mut self, event: &Value) -> Result<(), KernelError> {
        let id = event
            .get("call_id")
            .and_then(Value::as_str)
            .filter(|value| !value.is_empty())
            .ok_or_else(|| {
                KernelError::Model("OpenAI Responses function delta omitted call_id".into())
            })?;
        let delta = event.get("delta").and_then(Value::as_str).ok_or_else(|| {
            KernelError::Model("OpenAI Responses function delta omitted text".into())
        })?;
        let call = self
            .tools
            .entry(id.into())
            .or_insert_with(|| ResponsesToolCallBuffer {
                id: id.into(),
                name: String::new(),
                arguments: String::new(),
                started: false,
            });
        call.arguments.push_str(delta);
        if call.started {
            self.pending.push_back(ModelEvent::ToolArgumentsDelta {
                id: call.id.clone(),
                json_fragment: delta.into(),
            });
        }
        Ok(())
    }

    fn complete(&mut self, event: &Value) -> Result<(), KernelError> {
        if self.completed {
            return Ok(());
        }
        let response_value = event.get("response").ok_or_else(|| {
            KernelError::Model("OpenAI Responses completion omitted response".into())
        })?;
        let mut response = parse_responses_response(response_value)?;
        if response.content.is_empty() {
            response.content = self.content.clone();
        }
        if let Some(usage) = response_value.get("usage") {
            self.pending.push_back(ModelEvent::Usage {
                input_tokens: usage
                    .get("input_tokens")
                    .and_then(Value::as_u64)
                    .unwrap_or(0),
                output_tokens: usage
                    .get("output_tokens")
                    .and_then(Value::as_u64)
                    .unwrap_or(0),
            });
        }
        for call in &response.tool_calls {
            self.pending
                .push_back(ModelEvent::ToolCallReady { call: call.clone() });
        }
        self.pending.push_back(ModelEvent::Completed { response });
        self.completed = true;
        Ok(())
    }
}

fn parse_responses_response(response: &Value) -> Result<ModelResponse, KernelError> {
    let output = response
        .get("output")
        .and_then(Value::as_array)
        .ok_or_else(|| KernelError::Model("OpenAI Responses completion omitted output".into()))?;
    let mut content = Vec::new();
    let mut tool_calls = Vec::new();
    for item in output {
        match item.get("type").and_then(Value::as_str) {
            Some("message") => {
                if let Some(parts) = item.get("content").and_then(Value::as_array) {
                    content.extend(parts.iter().filter_map(|part| {
                        (part.get("type").and_then(Value::as_str) == Some("output_text"))
                            .then(|| part.get("text").and_then(Value::as_str))
                            .flatten()
                            .map(str::to_owned)
                    }));
                }
            }
            Some("function_call") => {
                let id = item
                    .get("call_id")
                    .or_else(|| item.get("id"))
                    .and_then(Value::as_str)
                    .filter(|value| !value.is_empty())
                    .ok_or_else(|| {
                        KernelError::Model("OpenAI Responses function call omitted call_id".into())
                    })?;
                let name = item
                    .get("name")
                    .and_then(Value::as_str)
                    .filter(|value| !value.is_empty())
                    .ok_or_else(|| {
                        KernelError::Model("OpenAI Responses function call omitted name".into())
                    })?;
                let arguments = match item.get("arguments") {
                    Some(Value::String(arguments)) => {
                        serde_json::from_str(arguments).map_err(|error| {
                            KernelError::Model(format!(
                                "OpenAI Responses function arguments were invalid JSON: {error}"
                            ))
                        })?
                    }
                    Some(arguments @ Value::Object(_)) => arguments.clone(),
                    _ => {
                        return Err(KernelError::Model(
                            "OpenAI Responses function call omitted arguments".into(),
                        ));
                    }
                };
                tool_calls.push(ToolCall {
                    id: id.into(),
                    name: name.into(),
                    arguments,
                });
            }
            _ => {}
        }
    }
    Ok(ModelResponse {
        content: content.join("\n"),
        tool_calls,
    })
}

#[derive(Default)]
struct OpenAiToolCallBuffer {
    id: String,
    name: String,
    arguments: String,
    started: bool,
}

struct OpenAiSseDecoder {
    api_key: Option<String>,
    max_response_bytes: usize,
    received_bytes: usize,
    buffer: Vec<u8>,
    pending: VecDeque<ModelEvent>,
    content: String,
    tools: BTreeMap<usize, OpenAiToolCallBuffer>,
    terminal: bool,
}

impl OpenAiSseDecoder {
    fn new(
        provider: String,
        model: String,
        endpoint: String,
        api_key: Option<String>,
        max_response_bytes: usize,
    ) -> Self {
        Self {
            api_key,
            max_response_bytes,
            received_bytes: 0,
            buffer: Vec::new(),
            pending: VecDeque::from([ModelEvent::RequestStarted {
                provider,
                model,
                endpoint,
            }]),
            content: String::new(),
            tools: BTreeMap::new(),
            terminal: false,
        }
    }

    fn push_chunk(&mut self, chunk: &[u8]) {
        if self.terminal {
            return;
        }
        self.received_bytes = self.received_bytes.saturating_add(chunk.len());
        if self.received_bytes > self.max_response_bytes {
            self.fail(KernelError::Model(format!(
                "OpenAI response exceeded the {} byte limit",
                self.max_response_bytes
            )));
            return;
        }
        self.buffer.extend_from_slice(chunk);
        while let Some((frame_end, delimiter_len)) = sse_frame_boundary(&self.buffer) {
            let frame = self.buffer.drain(..frame_end).collect::<Vec<_>>();
            self.buffer.drain(..delimiter_len);
            if let Err(error) = self.consume_frame(&frame) {
                self.fail(error);
                return;
            }
            if self.terminal {
                return;
            }
        }
    }

    fn can_retry_without_duplicate_output(&self) -> bool {
        self.content.trim().is_empty() && self.tools.is_empty()
    }

    fn consume_frame(&mut self, frame: &[u8]) -> Result<(), KernelError> {
        let frame = std::str::from_utf8(frame).map_err(|error| {
            KernelError::Model(format!("OpenAI SSE frame was not UTF-8: {error}"))
        })?;
        let data = frame
            .lines()
            .filter_map(|line| line.strip_prefix("data:"))
            .map(|line| line.strip_prefix(' ').unwrap_or(line))
            .collect::<Vec<_>>();
        if data.is_empty() {
            return Ok(());
        }
        let data = data.join("\n");
        if data == "[DONE]" {
            return self.complete();
        }
        let value: Value = serde_json::from_str(&data)
            .map_err(|error| KernelError::Model(format!("invalid OpenAI SSE JSON: {error}")))?;
        if let Some(error) = value.get("error") {
            return Err(KernelError::Model(format!("OpenAI SSE error: {error}")));
        }
        let choice = value
            .get("choices")
            .and_then(Value::as_array)
            .and_then(|choices| choices.first());
        if let Some(delta) = choice.and_then(|choice| choice.get("delta")) {
            if let Some(summary) = delta
                .get("reasoning_content")
                .or_else(|| delta.get("reasoning"))
                .and_then(Value::as_str)
                .filter(|summary| !summary.is_empty())
            {
                self.pending.push_back(ModelEvent::ReasoningSummaryDelta {
                    text: summary.into(),
                });
            }
            if let Some(text) = delta.get("content").and_then(Value::as_str) {
                self.content.push_str(text);
                self.pending
                    .push_back(ModelEvent::TextDelta { text: text.into() });
            }
            if let Some(calls) = delta.get("tool_calls").and_then(Value::as_array) {
                for call in calls {
                    self.consume_tool_delta(call)?;
                }
            }
        }
        if let Some(usage) = value.get("usage") {
            let input_tokens = usage
                .get("prompt_tokens")
                .or_else(|| usage.get("input_tokens"))
                .and_then(Value::as_u64)
                .unwrap_or(0);
            let output_tokens = usage
                .get("completion_tokens")
                .or_else(|| usage.get("output_tokens"))
                .and_then(Value::as_u64)
                .unwrap_or(0);
            self.pending.push_back(ModelEvent::Usage {
                input_tokens,
                output_tokens,
            });
        }
        if choice
            .and_then(|choice| choice.get("finish_reason"))
            .is_some_and(|reason| !reason.is_null())
        {
            self.complete()?;
        }
        Ok(())
    }

    fn consume_tool_delta(&mut self, value: &Value) -> Result<(), KernelError> {
        let index = value
            .get("index")
            .and_then(Value::as_u64)
            .ok_or_else(|| KernelError::Model("OpenAI SSE tool call omitted index".into()))?
            as usize;
        let id = value.get("id").and_then(Value::as_str);
        let function = value.get("function").unwrap_or(&Value::Null);
        let name = function.get("name").and_then(Value::as_str);
        let arguments = function.get("arguments").and_then(Value::as_str);
        let call = self.tools.entry(index).or_default();
        if let Some(id) = id.filter(|id| !id.is_empty()) {
            call.id = id.into();
        }
        if let Some(name) = name.filter(|name| !name.is_empty()) {
            call.name = name.into();
        }
        if !call.started && !call.id.is_empty() && !call.name.is_empty() {
            call.started = true;
            self.pending.push_back(ModelEvent::ToolCallStarted {
                id: call.id.clone(),
                name: call.name.clone(),
            });
        }
        if let Some(arguments) = arguments {
            call.arguments.push_str(arguments);
            if call.started {
                self.pending.push_back(ModelEvent::ToolArgumentsDelta {
                    id: call.id.clone(),
                    json_fragment: arguments.into(),
                });
            }
        }
        Ok(())
    }

    fn complete(&mut self) -> Result<(), KernelError> {
        if self.terminal {
            return Ok(());
        }
        let mut tool_calls = Vec::with_capacity(self.tools.len());
        for call in self.tools.values() {
            if call.id.is_empty() || call.name.is_empty() {
                return Err(KernelError::Model(
                    "OpenAI SSE tool call completed without id or name".into(),
                ));
            }
            let arguments = serde_json::from_str(&call.arguments).map_err(|error| {
                KernelError::Model(format!(
                    "OpenAI SSE tool arguments were invalid JSON: {error}"
                ))
            })?;
            let call = ToolCall {
                id: call.id.clone(),
                name: call.name.clone(),
                arguments,
            };
            self.pending
                .push_back(ModelEvent::ToolCallReady { call: call.clone() });
            tool_calls.push(call);
        }
        self.pending.push_back(ModelEvent::Completed {
            response: ModelResponse {
                content: self.content.clone(),
                tool_calls,
            },
        });
        self.terminal = true;
        Ok(())
    }

    fn finish_eof(&mut self) {
        if !self.terminal {
            self.fail(KernelError::Model(
                "OpenAI SSE stream ended without a terminal event".into(),
            ));
        }
    }

    fn cancel(&mut self) {
        if !self.terminal {
            self.pending.push_back(ModelEvent::Cancelled);
            self.terminal = true;
        }
    }

    fn fail(&mut self, error: KernelError) {
        if !self.terminal {
            self.pending.push_back(ModelEvent::Failed {
                error: redact_secret(error.to_string(), self.api_key.as_deref()),
            });
            self.terminal = true;
        }
    }
}

fn sse_frame_boundary(bytes: &[u8]) -> Option<(usize, usize)> {
    bytes
        .windows(4)
        .position(|window| window == b"\r\n\r\n")
        .map(|index| (index, 4))
        .or_else(|| {
            bytes
                .windows(2)
                .position(|window| window == b"\n\n")
                .map(|index| (index, 2))
        })
}

fn openai_endpoint(base_url: &str, wire_api: OpenAiWireApi) -> Result<reqwest::Url, KernelError> {
    let mut endpoint = reqwest::Url::parse(base_url.trim())
        .map_err(|error| KernelError::Model(format!("invalid OpenAI base URL: {error}")))?;
    if endpoint.query().is_some() || endpoint.fragment().is_some() {
        return Err(KernelError::Model(
            "OpenAI base URL must not contain a query or fragment".into(),
        ));
    }
    let base_path = endpoint.path().trim_end_matches('/');
    let api_path = if base_path.is_empty() {
        "/v1"
    } else {
        base_path
    };
    let resource = match wire_api {
        OpenAiWireApi::ChatCompletions => "chat/completions",
        OpenAiWireApi::Responses => "responses",
    };
    endpoint.set_path(&format!("{api_path}/{resource}"));
    Ok(endpoint)
}

fn openai_response_error(
    status: u16,
    content_type: &str,
    bytes: &[u8],
    detail: &str,
    api_key: Option<&str>,
) -> KernelError {
    KernelError::Model(redact_secret(
        format!(
            "OpenAI response rejected: status={status} content-type={content_type} detail={detail} body={:?}",
            bounded_http_body(bytes)
        ),
        api_key,
    ))
}

fn bounded_http_body(bytes: &[u8]) -> String {
    let retained = &bytes[..bytes.len().min(HTTP_RESPONSE_DIAGNOSTIC_BYTES)];
    let mut body = String::from_utf8_lossy(retained).into_owned();
    if retained.len() < bytes.len() {
        body.push_str("...[truncated]");
    }
    body
}

fn openai_message(message: &Message) -> Value {
    match message.role {
        Role::System => json!({"role":"system","content":message.content}),
        Role::User => json!({"role":"user","content":message.content}),
        Role::Assistant => {
            let calls = message
                .tool_calls
                .iter()
                .map(|call| {
                    json!({
                        "id": call.id,
                        "type": "function",
                        "function": {"name": call.name, "arguments": call.arguments.to_string()}
                    })
                })
                .collect::<Vec<_>>();
            let content = if message.content.is_empty() && !calls.is_empty() {
                Value::Null
            } else {
                Value::String(message.content.clone())
            };
            if calls.is_empty() {
                json!({"role":"assistant","content":content})
            } else {
                json!({"role":"assistant","content":content,"tool_calls":calls})
            }
        }
        Role::Tool => json!({
            "role":"tool",
            "tool_call_id": message.tool_call_id,
            "name": message.tool_name,
            "content": message.content,
        }),
    }
}

fn openai_messages(messages: &[Message]) -> Vec<Value> {
    messages
        .iter()
        .filter(|message| {
            !(message.role == Role::Assistant
                && message.content.is_empty()
                && message.tool_calls.is_empty())
        })
        .map(openai_message)
        .collect()
}

fn openai_responses_request(request: &ModelRequest, model: &str) -> Value {
    let mut instructions = Vec::new();
    let mut input = Vec::new();
    for message in &request.messages {
        match message.role {
            Role::System => instructions.push(message.content.as_str()),
            Role::User => input.push(json!({
                "role": "user",
                "content": [{"type": "input_text", "text": message.content}],
            })),
            Role::Assistant => {
                if !message.content.is_empty() {
                    input.push(json!({
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": message.content}],
                    }));
                }
                input.extend(message.tool_calls.iter().map(|call| {
                    json!({
                        "type": "function_call",
                        "call_id": call.id,
                        "name": call.name,
                        "arguments": call.arguments.to_string(),
                    })
                }));
            }
            Role::Tool => input.push(json!({
                "type": "function_call_output",
                "call_id": message.tool_call_id,
                "output": message.content,
            })),
        }
    }
    let tools = request
        .tools
        .iter()
        .map(|tool| {
            json!({
                "type": "function",
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.input_schema,
                "strict": false,
            })
        })
        .collect::<Vec<_>>();
    let mut body = json!({
        "model": model,
        "stream": true,
        "input": input,
        "tools": tools,
    });
    if !instructions.is_empty() {
        body["instructions"] = Value::String(instructions.join("\n\n"));
    }
    body
}

fn parse_openai_response(bytes: &[u8]) -> Result<ModelResponse, KernelError> {
    let value: Value = serde_json::from_slice(bytes)
        .map_err(|error| KernelError::Model(format!("invalid OpenAI response JSON: {error}")))?;
    let message = value
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|choices| choices.first())
        .and_then(|choice| choice.get("message"))
        .ok_or_else(|| KernelError::Model("OpenAI response contained no choices".into()))?;
    let content = parse_content(message.get("content"))?;
    let tool_calls = message
        .get("tool_calls")
        .and_then(Value::as_array)
        .map(|calls| calls.iter().map(parse_tool_call).collect())
        .transpose()?
        .unwrap_or_default();
    Ok(ModelResponse {
        content,
        tool_calls,
    })
}

fn parse_content(content: Option<&Value>) -> Result<String, KernelError> {
    match content {
        None | Some(Value::Null) => Ok(String::new()),
        Some(Value::String(content)) => Ok(content.clone()),
        Some(Value::Array(blocks)) => {
            let mut output = String::new();
            let mut seen_text = false;
            for text in blocks
                .iter()
                .filter_map(|block| block.get("text").and_then(Value::as_str))
            {
                if seen_text {
                    output.push('\n');
                }
                output.push_str(text);
                seen_text = true;
            }
            Ok(output)
        }
        Some(_) => Err(KernelError::Model(
            "OpenAI assistant content had an unsupported shape".into(),
        )),
    }
}

fn parse_tool_call(value: &Value) -> Result<ToolCall, KernelError> {
    let id = value
        .get("id")
        .and_then(Value::as_str)
        .ok_or_else(|| KernelError::Model("OpenAI tool call omitted id".into()))?;
    let function = value
        .get("function")
        .ok_or_else(|| KernelError::Model("OpenAI tool call omitted function".into()))?;
    let name = function
        .get("name")
        .and_then(Value::as_str)
        .ok_or_else(|| KernelError::Model("OpenAI tool call omitted function name".into()))?;
    let arguments = match function.get("arguments") {
        Some(Value::String(arguments)) => serde_json::from_str(arguments).map_err(|error| {
            KernelError::Model(format!("OpenAI tool arguments were invalid JSON: {error}"))
        })?,
        Some(arguments @ Value::Object(_)) => arguments.clone(),
        _ => {
            return Err(KernelError::Model(
                "OpenAI tool arguments must be a JSON string or object".into(),
            ));
        }
    };
    Ok(ToolCall {
        id: id.into(),
        name: name.into(),
        arguments,
    })
}

fn redact_secret(mut message: String, secret: Option<&str>) -> String {
    if let Some(secret) = secret.filter(|secret| !secret.is_empty()) {
        message = message.replace(secret, "[redacted]");
    }
    message
}

/// A provider adapter command. It reads a [`ModelRequest`] JSON document from stdin
/// and writes one [`ModelResponse`] JSON document to stdout.
#[derive(Debug, Clone)]
pub struct CommandModelProvider {
    program: PathBuf,
    arguments: Vec<String>,
    timeout: Duration,
    stdout_limit: usize,
    stderr_limit: usize,
}

impl CommandModelProvider {
    /// Create a provider adapter without invoking a shell.
    #[must_use]
    pub fn new(program: impl Into<PathBuf>, arguments: Vec<String>) -> Self {
        Self {
            program: program.into(),
            arguments,
            timeout: DEFAULT_COMMAND_TIMEOUT,
            stdout_limit: DEFAULT_COMMAND_OUTPUT_LIMIT,
            stderr_limit: DEFAULT_COMMAND_OUTPUT_LIMIT,
        }
    }

    /// Set the wall-clock deadline covering process execution and pipe drain.
    #[must_use]
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Set independent retained byte limits for adapter stdout and stderr.
    #[must_use]
    pub fn with_output_limits(mut self, stdout_limit: usize, stderr_limit: usize) -> Self {
        self.stdout_limit = stdout_limit.max(1);
        self.stderr_limit = stderr_limit.max(1);
        self
    }
}

#[async_trait::async_trait]
impl ModelProvider for CommandModelProvider {
    async fn stream(
        &self,
        request: ModelRequest,
        cancellation: &dyn CancellationSignal,
    ) -> Result<ModelEventStream, KernelError> {
        if cancellation.is_cancelled() {
            return Ok(model_event_stream(vec![ModelEvent::Cancelled]));
        }
        let cancellation = CancellationBridge::from_signal(cancellation);
        let provider = self.clone();
        let worker_cancellation = cancellation.clone();
        let (sender, receiver) = futures::channel::mpsc::unbounded();
        let _ = sender.unbounded_send(Ok(ModelEvent::RequestStarted {
            provider: "command-adapter".into(),
            model: "command-adapter".into(),
            endpoint: "command://adapter".into(),
        }));

        tokio::task::spawn_blocking(move || {
            let events = match provider.complete_request(request, &worker_cancellation) {
                Ok(response) => command_completed_events(response),
                Err(KernelError::Cancelled) => vec![ModelEvent::Cancelled],
                Err(error) => vec![ModelEvent::Failed {
                    error: error.to_string(),
                }],
            };
            for event in events {
                if sender.unbounded_send(Ok(event)).is_err() {
                    break;
                }
            }
        });
        Ok(Box::pin(receiver))
    }
}

impl CommandModelProvider {
    fn complete_request(
        &self,
        request: ModelRequest,
        cancellation: &dyn CancellationSignal,
    ) -> Result<ModelResponse, KernelError> {
        if cancellation.is_cancelled() {
            return Err(KernelError::Cancelled);
        }
        let payload =
            serde_json::to_vec(&request).map_err(|error| KernelError::Model(error.to_string()))?;
        let mut command = Command::new(&self.program);
        command
            .args(&self.arguments)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        let mut process = command
            .group_spawn()
            .map_err(|error| KernelError::Model(format!("adapter start failed: {error}")))?;
        let stdout = process
            .inner()
            .stdout
            .take()
            .ok_or_else(|| KernelError::Model("adapter stdout was not available".into()))?;
        let stderr = process
            .inner()
            .stderr
            .take()
            .ok_or_else(|| KernelError::Model("adapter stderr was not available".into()))?;
        let stdout_limit = self.stdout_limit;
        let stderr_limit = self.stderr_limit;
        let stdout_reader = thread::spawn(move || read_bounded(stdout, stdout_limit));
        let stderr_reader = thread::spawn(move || read_bounded(stderr, stderr_limit));
        let input_result = process
            .inner()
            .stdin
            .take()
            .ok_or_else(|| KernelError::Model("adapter stdin was not available".into()))?
            .write_all(&payload)
            .map_err(|error| KernelError::Model(format!("adapter input failed: {error}")));
        if let Err(error) = input_result {
            terminate_adapter_group(&mut process);
            let _ = join_reader(stdout_reader, "stdout");
            let _ = join_reader(stderr_reader, "stderr");
            return Err(error);
        }
        let started = Instant::now();
        let mut direct_status = None;
        let status = loop {
            if cancellation.is_cancelled() {
                terminate_adapter_group(&mut process);
                let _ = join_reader(stdout_reader, "stdout");
                let _ = join_reader(stderr_reader, "stderr");
                return Err(KernelError::Cancelled);
            }
            if started.elapsed() >= self.timeout {
                terminate_adapter_group(&mut process);
                let _ = join_reader(stdout_reader, "stdout");
                let _ = join_reader(stderr_reader, "stderr");
                return Err(KernelError::Model(format!(
                    "adapter timed out after {:?}",
                    self.timeout
                )));
            }
            if direct_status.is_none() {
                match process.try_wait() {
                    Ok(Some(status)) => direct_status = Some(status),
                    Ok(None) => {}
                    Err(error) => {
                        terminate_adapter_group(&mut process);
                        let _ = join_reader(stdout_reader, "stdout");
                        let _ = join_reader(stderr_reader, "stderr");
                        return Err(KernelError::Model(format!("adapter wait failed: {error}")));
                    }
                }
            }
            if stdout_reader.is_finished()
                && stderr_reader.is_finished()
                && let Some(status) = direct_status
            {
                break status;
            }
            thread::sleep(Duration::from_millis(10));
        };
        let stdout = join_reader(stdout_reader, "stdout")?;
        let stderr = join_reader(stderr_reader, "stderr")?;
        if stdout.truncated {
            return Err(KernelError::Model(format!(
                "adapter stdout exceeded the {} byte limit and was truncated",
                self.stdout_limit
            )));
        }
        if !status.success() {
            return Err(KernelError::Model(format!(
                "adapter exited with {:?}: {}",
                status.code(),
                String::from_utf8_lossy(&stderr.bytes)
            )));
        }
        serde_json::from_slice(&stdout.bytes).map_err(|error| {
            KernelError::Model(format!(
                "adapter emitted invalid ModelResponse JSON: {error}; stderr: {}",
                String::from_utf8_lossy(&stderr.bytes)
            ))
        })
    }
}

fn command_completed_events(response: ModelResponse) -> Vec<ModelEvent> {
    let mut events = Vec::new();
    if !response.content.is_empty() {
        events.push(ModelEvent::TextDelta {
            text: response.content.clone(),
        });
    }
    for call in &response.tool_calls {
        events.push(ModelEvent::ToolCallStarted {
            id: call.id.clone(),
            name: call.name.clone(),
        });
        events.push(ModelEvent::ToolCallReady { call: call.clone() });
    }
    events.push(ModelEvent::Completed { response });
    events
}

pub(crate) struct BoundedBytes {
    pub(crate) bytes: Vec<u8>,
    pub(crate) truncated: bool,
}

pub(crate) fn read_bounded(mut reader: impl Read, limit: usize) -> std::io::Result<BoundedBytes> {
    let mut bytes = Vec::with_capacity(limit.min(64 * 1024));
    let mut truncated = false;
    let mut chunk = [0_u8; 8 * 1024];
    loop {
        let read = reader.read(&mut chunk)?;
        if read == 0 {
            break;
        }
        let remaining = limit.saturating_sub(bytes.len());
        let keep = remaining.min(read);
        bytes.extend_from_slice(&chunk[..keep]);
        truncated |= keep < read;
    }
    Ok(BoundedBytes { bytes, truncated })
}

fn join_reader(
    reader: thread::JoinHandle<std::io::Result<BoundedBytes>>,
    stream: &str,
) -> Result<BoundedBytes, KernelError> {
    reader
        .join()
        .map_err(|_| KernelError::Model(format!("adapter {stream} reader panicked")))?
        .map_err(|error| KernelError::Model(format!("adapter {stream} read failed: {error}")))
}

fn terminate_adapter_group(process: &mut GroupChild) {
    let _ = process.kill();
    let _ = process.wait();
}

#[cfg(test)]
mod tests {
    use super::OpenAiSseDecoder;
    use std::{
        io::{Read, Write},
        net::TcpListener,
        sync::mpsc,
        thread,
        time::{Duration, Instant},
    };

    use focus_kernel::{
        KernelError, Message, ModelEvent, ModelProvider, ModelRequest, ModelResponse,
        NoCancellation, Role, ToolCall, ToolDefinition,
    };
    use serde_json::{Value, json};

    use crate::{mcp::test_support::wait_for_process_exit, subagent::CancellationToken};

    use super::{
        CommandModelProvider, DEFAULT_OPENAI_RETRY_BASE_DELAY, OpenAiCompatibleProvider,
        OpenAiConfig, OpenAiWireApi, parse_content, retry_delay,
    };

    fn collect_response(
        provider: &dyn ModelProvider,
        request: ModelRequest,
    ) -> Result<ModelResponse, KernelError> {
        use futures::TryStreamExt;

        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(async {
                let events = provider
                    .stream(request, &NoCancellation)
                    .await?
                    .try_collect::<Vec<_>>()
                    .await?;
                events
                    .into_iter()
                    .rev()
                    .find_map(|event| match event {
                        ModelEvent::Completed { response } => Some(Ok(response)),
                        ModelEvent::Failed { error } => Some(Err(KernelError::Model(error))),
                        ModelEvent::Cancelled => Some(Err(KernelError::Cancelled)),
                        _ => None,
                    })
                    .unwrap_or_else(|| {
                        Err(KernelError::Model(
                            "model stream ended without completion".into(),
                        ))
                    })
            })
    }

    #[test]
    fn parses_array_content_without_changing_empty_blocks() {
        let content = parse_content(Some(&json!([
            {"text": ""},
            {"type": "image"},
            {"text": "second"},
        ])))
        .unwrap();

        assert_eq!(content, "\nsecond");
    }

    #[test]
    fn command_provider_times_out_and_bounds_adapter_output() {
        let _process_guard = crate::sandbox::process_test_lock().lock().unwrap();
        let helper = std::env::current_exe().unwrap();
        let provider = CommandModelProvider::new(
            helper,
            vec![
                "--exact".into(),
                "provider::tests::command_provider_timeout_helper".into(),
                "--ignored".into(),
                "--nocapture".into(),
            ],
        )
        .with_timeout(Duration::from_millis(150))
        .with_output_limits(128, 128);

        let started = Instant::now();
        let error = provider
            .complete_request(
                ModelRequest {
                    messages: vec![Message::text(Role::User, "hello")],
                    tools: Vec::new(),
                },
                &NoCancellation,
            )
            .unwrap_err()
            .to_string();

        assert!(error.contains("timed out"), "{error}");
        assert!(error.len() < 512, "adapter error was {} bytes", error.len());
        // Windows process-group teardown can briefly contend with workspace
        // test workers; keep a bounded upper limit without asserting a
        // scheduler-specific one-second startup budget.
        assert!(started.elapsed() < Duration::from_secs(5));
    }

    #[test]
    fn command_provider_rejects_bounded_truncated_output() {
        let helper = std::env::current_exe().unwrap();
        let provider = CommandModelProvider::new(
            helper,
            vec![
                "--exact".into(),
                "provider::tests::command_provider_output_helper".into(),
                "--ignored".into(),
                "--nocapture".into(),
            ],
        )
        .with_output_limits(128, 128);

        let error = provider
            .complete_request(
                ModelRequest {
                    messages: vec![Message::text(Role::User, "hello")],
                    tools: Vec::new(),
                },
                &NoCancellation,
            )
            .unwrap_err()
            .to_string();

        assert!(error.contains("truncated"), "{error}");
        assert!(
            error.len() < 2_000,
            "adapter error was {} bytes",
            error.len()
        );
    }

    #[test]
    fn command_provider_stream_emits_lifecycle_events_from_its_single_response() {
        let (program, arguments) = if cfg!(windows) {
            (
                "python.exe",
                vec![
                    "-c".into(),
                    "import json; print(json.dumps({'content': 'command result', 'tool_calls': []}))"
                        .into(),
                ],
            )
        } else {
            (
                "sh",
                vec![
                    "-c".into(),
                    "printf '%s\\n' '{\"content\":\"command result\",\"tool_calls\":[]}'".into(),
                ],
            )
        };
        let provider = CommandModelProvider::new(program, arguments);

        let events = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(async {
                use futures::TryStreamExt;

                provider
                    .stream(
                        ModelRequest {
                            messages: vec![Message::text(Role::User, "hello")],
                            tools: Vec::new(),
                        },
                        &NoCancellation,
                    )
                    .await
                    .unwrap()
                    .try_collect::<Vec<_>>()
                    .await
                    .unwrap()
            });

        assert!(matches!(
            events.first(),
            Some(ModelEvent::RequestStarted { provider, .. }) if provider == "command-adapter"
        ));
        assert!(matches!(
            events.get(1),
            Some(ModelEvent::TextDelta { text }) if text == "command result"
        ));
        assert!(matches!(
            events.last(),
            Some(ModelEvent::Completed { response }) if response.content == "command result"
        ));
    }

    #[test]
    #[ignore = "subprocess helper for command provider timeout coverage"]
    fn command_provider_timeout_helper() {
        println!("{}", "x".repeat(100_000));
        thread::sleep(Duration::from_secs(10));
    }

    #[test]
    #[ignore = "subprocess helper for command provider output coverage"]
    fn command_provider_output_helper() {
        println!("{}", "x".repeat(100_000));
    }

    #[test]
    fn openai_provider_preserves_tool_call_transcript_and_parses_response_calls() {
        let response = json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "working",
                    "tool_calls": [{
                        "id": "call-next",
                        "type": "function",
                        "function": {"name": "search", "arguments": {"query": "needle"}}
                    }]
                }
            }]
        });
        let (base_url, captured, server) = fake_http_server(200, response);
        let provider = OpenAiCompatibleProvider::new(
            OpenAiConfig::new(base_url, "gpt-test").with_api_key("secret-token"),
        )
        .unwrap();
        let request = ModelRequest {
            messages: vec![
                Message::text(Role::System, "system"),
                Message::assistant(
                    "",
                    vec![ToolCall {
                        id: "call-1".into(),
                        name: "read_file".into(),
                        arguments: json!({"path":"README.md"}),
                    }],
                ),
                Message::tool_result("call-1", "read_file", "contents"),
            ],
            tools: vec![ToolDefinition {
                name: "search".into(),
                description: "search files".into(),
                input_schema: json!({"type":"object"}),
            }],
        };

        let result = collect_response(&provider, request).unwrap();
        let raw = captured.recv().unwrap();
        server.join().unwrap();
        let (headers, body) = split_http_request(&raw);
        let body: Value = serde_json::from_str(body).unwrap();

        assert!(headers.contains("authorization: Bearer secret-token"));
        assert_eq!(body["model"], "gpt-test");
        assert_eq!(body["messages"][1]["tool_calls"][0]["id"], "call-1");
        assert_eq!(body["messages"][2]["tool_call_id"], "call-1");
        assert_eq!(body["tools"][0]["function"]["name"], "search");
        assert_eq!(result.content, "working");
        assert_eq!(result.tool_calls[0].arguments, json!({"query":"needle"}));
    }

    #[test]
    fn openai_message_uses_empty_string_for_assistant_without_tool_calls() {
        let message = super::openai_message(&Message::assistant("", Vec::new()));

        assert_eq!(message["role"], "assistant");
        assert_eq!(message["content"], "");
        assert!(message.get("tool_calls").is_none());
    }

    #[test]
    fn openai_request_omits_empty_assistant_without_tool_calls() {
        let messages = super::openai_messages(&[
            Message::text(Role::User, "task"),
            Message::assistant("", Vec::new()),
            Message::text(Role::System, "continue"),
        ]);

        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0]["role"], "user");
        assert_eq!(messages[1]["role"], "system");
    }

    #[test]
    fn openai_stream_yields_text_tool_usage_and_completion_before_body_close() {
        use futures::{StreamExt, TryStreamExt};

        let frames = vec![
            "data: {\"choices\":[{\"delta\":{\"content\":\"hel\"}}]}\n\n".into(),
            "data: {\"choices\":[{\"delta\":{\"content\":\"lo\"}}]}\n\n".into(),
            format!(
                "data: {}\n\n",
                json!({"choices":[{"delta":{"tool_calls":[{
                    "index":0,
                    "id":"call-1",
                    "type":"function",
                    "function":{"name":"search","arguments":"{\"query\":"}
                }]}}]})
            ),
            format!(
                "data: {}\n\n",
                json!({
                    "choices":[{"delta":{"tool_calls":[{
                        "index":0,
                        "id":"",
                        "function":{"name":"","arguments":"\"needle\"}"}
                    }]}}],
                    "usage":{"prompt_tokens":3,"completion_tokens":2}
                })
            ),
            "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n".into(),
            "data: [DONE]\n\n".into(),
        ];
        let (base_url, captured, server) = sse_http_server(frames, Duration::from_millis(75));
        let provider =
            OpenAiCompatibleProvider::new(OpenAiConfig::new(base_url, "gpt-test")).unwrap();
        let started = Instant::now();
        let (first, first_elapsed, events) = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(async {
                let mut stream = provider
                    .stream(
                        ModelRequest {
                            messages: vec![Message::text(Role::User, "hello")],
                            tools: Vec::new(),
                        },
                        &NoCancellation,
                    )
                    .await
                    .unwrap();
                let first = stream.next().await.unwrap().unwrap();
                let first_elapsed = started.elapsed();
                let mut events = vec![first.clone()];
                events.extend(stream.try_collect::<Vec<_>>().await.unwrap());
                (first, first_elapsed, events)
            });
        let request = captured.recv().unwrap();
        server.join().unwrap();

        assert!(first_elapsed < Duration::from_millis(300));
        assert!(matches!(first, ModelEvent::RequestStarted { .. }));
        assert!(matches!(events[1], ModelEvent::TextDelta { ref text } if text == "hel"));
        assert!(matches!(events[2], ModelEvent::TextDelta { ref text } if text == "lo"));
        assert!(events.iter().any(|event| matches!(
            event,
            ModelEvent::ToolArgumentsDelta { json_fragment, .. } if json_fragment == "{\"query\":"
        )));
        assert!(events.iter().all(|event| !matches!(
            event,
            ModelEvent::ToolArgumentsDelta { id, .. } if id != "call-1"
        )));
        assert!(
            events.iter().any(|event| matches!(
                event,
                ModelEvent::ToolCallReady { call }
                    if call.name == "search" && call.arguments == json!({"query":"needle"})
            )),
            "events: {events:#?}"
        );
        assert!(events.iter().any(|event| matches!(
            event,
            ModelEvent::Usage {
                input_tokens: 3,
                output_tokens: 2
            }
        )));
        assert!(
            matches!(events.last(), Some(ModelEvent::Completed { response }) if response.content == "hello")
        );
        let (_, body) = split_http_request(&request);
        assert!(
            serde_json::from_str::<Value>(body).unwrap()["stream"]
                .as_bool()
                .unwrap()
        );
    }

    #[test]
    fn chat_provider_retries_an_empty_completion_before_exposing_a_terminal_event() {
        use futures::TryStreamExt;

        let empty = vec![
            "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n".into(),
            "data: [DONE]\n\n".into(),
        ];
        let recovered = vec![
            "data: {\"choices\":[{\"delta\":{\"content\":\"recovered\"}}]}\n\n".into(),
            "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n".into(),
            "data: [DONE]\n\n".into(),
        ];
        let (base_url, requests, server) =
            response_sse_server(vec![(Vec::new(), empty), (Vec::new(), recovered)]);
        let provider = OpenAiCompatibleProvider::new(
            OpenAiConfig::new(base_url, "gpt-test")
                .with_max_retries(1)
                .with_retry_base_delay(Duration::from_millis(1)),
        )
        .unwrap();

        let events = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(async {
                provider
                    .stream(
                        ModelRequest {
                            messages: vec![Message::text(Role::User, "hello")],
                            tools: Vec::new(),
                        },
                        &NoCancellation,
                    )
                    .await
                    .unwrap()
                    .try_collect::<Vec<_>>()
                    .await
                    .unwrap()
            });
        let first = requests.recv().unwrap();
        let second = requests.recv().unwrap();
        server.join().unwrap();

        assert!(first.starts_with("POST /v1/chat/completions HTTP/1.1"));
        assert!(second.starts_with("POST /v1/chat/completions HTTP/1.1"));
        let first_headers = first.split_once("\r\n\r\n").unwrap().0;
        let second_headers = second.split_once("\r\n\r\n").unwrap().0;
        let first_request_id = first_headers
            .lines()
            .find_map(|line| line.strip_prefix("x-client-request-id: "))
            .unwrap();
        let second_request_id = second_headers
            .lines()
            .find_map(|line| line.strip_prefix("x-client-request-id: "))
            .unwrap();
        assert_eq!(first_request_id, second_request_id);
        assert!(events.iter().any(|event| matches!(
            event,
            ModelEvent::RetryScheduled { attempt: 1, reason, .. }
                if reason.contains("OpenAI stream completed without assistant text or tool calls")
        )));
        assert_eq!(
            events
                .iter()
                .filter(|event| matches!(event, ModelEvent::Completed { .. }))
                .count(),
            1
        );
        assert!(matches!(
            events.last(),
            Some(ModelEvent::Completed { response }) if response.content == "recovered"
        ));
    }

    #[test]
    fn responses_provider_retries_an_early_close_with_the_same_turn_identity() {
        use futures::TryStreamExt;

        let incomplete = vec![format!(
            "event: response.output_text.delta\ndata: {}\n\n",
            json!({
                "type": "response.output_text.delta",
                "response_id": "resp-first",
                "item_id": "msg-first",
                "delta": "partial"
            })
        )];
        let complete = vec![
            format!(
                "event: response.output_text.delta\ndata: {}\n\n",
                json!({
                    "type": "response.output_text.delta",
                    "response_id": "resp-second",
                    "item_id": "msg-second",
                    "delta": "recovered"
                })
            ),
            format!(
                "event: response.completed\ndata: {}\n\n",
                json!({
                    "type": "response.completed",
                    "response": {
                        "id": "resp-second",
                        "status": "completed",
                        "output": [{
                            "type": "message",
                            "id": "msg-second",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": "recovered"}]
                        }],
                        "usage": {"input_tokens": 3, "output_tokens": 1}
                    }
                })
            ),
        ];
        let (base_url, requests, server) = response_sse_server(vec![
            (
                vec![("x-codex-turn-state".into(), "turn-state-1".into())],
                incomplete,
            ),
            (Vec::new(), complete),
        ]);
        let provider = OpenAiCompatibleProvider::new(
            OpenAiConfig::new(base_url, "gpt-test")
                .with_wire_api(OpenAiWireApi::Responses)
                .with_max_retries(1)
                .with_retry_base_delay(Duration::from_millis(1)),
        )
        .unwrap();

        let events = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(async {
                provider
                    .stream(
                        ModelRequest {
                            messages: vec![Message::text(Role::User, "hello")],
                            tools: Vec::new(),
                        },
                        &NoCancellation,
                    )
                    .await
                    .unwrap()
                    .try_collect::<Vec<_>>()
                    .await
                    .unwrap()
            });
        let first = requests.recv().unwrap();
        let second = requests.recv().unwrap();
        server.join().unwrap();
        let (first_headers, first_body) = split_http_request(&first);
        let (second_headers, second_body) = split_http_request(&second);
        let first_request_id = first_headers
            .lines()
            .find_map(|line| line.strip_prefix("x-client-request-id: "))
            .unwrap();
        let second_request_id = second_headers
            .lines()
            .find_map(|line| line.strip_prefix("x-client-request-id: "))
            .unwrap();

        assert!(first.starts_with("POST /v1/responses HTTP/1.1"));
        assert!(second.starts_with("POST /v1/responses HTTP/1.1"));
        assert_eq!(first_request_id, second_request_id);
        assert!(second_headers.contains("x-codex-turn-state: turn-state-1"));
        assert_eq!(
            serde_json::from_str::<Value>(first_body).unwrap()["stream"],
            true
        );
        assert_eq!(
            serde_json::from_str::<Value>(second_body).unwrap()["stream"],
            true
        );
        assert!(
            events
                .iter()
                .any(|event| matches!(event, ModelEvent::RetryScheduled { attempt: 1, .. }))
        );
        assert_eq!(
            events
                .iter()
                .filter(|event| matches!(event, ModelEvent::Completed { .. }))
                .count(),
            1
        );
        assert!(matches!(
            events.last(),
            Some(ModelEvent::Completed { response }) if response.content == "recovered"
        ));
    }

    #[test]
    fn openai_provider_normalizes_an_unversioned_base_url() {
        let response = json!({"choices":[{"message":{"role":"assistant","content":"ok"}}]});
        let (versioned_base_url, captured, server) = fake_http_server(200, response);
        let base_url = versioned_base_url.trim_end_matches("/v1").to_owned();
        let provider =
            OpenAiCompatibleProvider::new(OpenAiConfig::new(base_url, "gpt-test")).unwrap();

        let result = collect_response(
            &provider,
            ModelRequest {
                messages: vec![Message::text(Role::User, "hello")],
                tools: Vec::new(),
            },
        )
        .unwrap();
        let request = captured.recv().unwrap();
        server.join().unwrap();

        assert!(request.starts_with("POST /v1/chat/completions HTTP/1.1"));
        assert_eq!(result.content, "ok");
    }

    #[test]
    fn openai_provider_retries_a_transient_status_and_preserves_attempt_lifecycle() {
        use futures::TryStreamExt;

        let (base_url, requests, server) = multi_response_http_server(vec![
            (503, json!({"error":{"message":"busy"}}).to_string()),
            (
                200,
                json!({"choices":[{"message":{"role":"assistant","content":"ok"}}]}).to_string(),
            ),
        ]);
        let provider = OpenAiCompatibleProvider::new(
            OpenAiConfig::new(base_url, "gpt-test")
                .with_max_retries(1)
                .with_retry_base_delay(Duration::from_millis(1)),
        )
        .unwrap();
        let events = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(async {
                provider
                    .stream(
                        ModelRequest {
                            messages: vec![Message::text(Role::User, "hello")],
                            tools: Vec::new(),
                        },
                        &NoCancellation,
                    )
                    .await
                    .unwrap()
                    .try_collect::<Vec<_>>()
                    .await
                    .unwrap()
            });
        let first = requests.recv().unwrap();
        let second = requests.recv().unwrap();
        server.join().unwrap();

        assert!(first.starts_with("POST /v1/chat/completions HTTP/1.1"));
        assert!(second.starts_with("POST /v1/chat/completions HTTP/1.1"));
        assert!(matches!(
            events.first(),
            Some(ModelEvent::RetryScheduled { attempt: 1, delay_ms: 1, reason })
                if reason == "HTTP status 503"
        ));
        assert!(matches!(
            events.get(1),
            Some(ModelEvent::RequestStarted { .. })
        ));
        assert!(
            matches!(events.last(), Some(ModelEvent::Completed { response }) if response.content == "ok")
        );
    }

    #[test]
    fn openai_provider_reports_scheduled_retry_before_cancellation() {
        use futures::TryStreamExt;

        let (base_url, _requests, server) = multi_response_http_server(vec![(
            503,
            json!({"error":{"message":"busy"}}).to_string(),
        )]);
        let provider = OpenAiCompatibleProvider::new(
            OpenAiConfig::new(base_url, "gpt-test")
                .with_max_retries(1)
                .with_retry_base_delay(Duration::from_millis(500)),
        )
        .unwrap();
        let token = CancellationToken::default();
        let trigger = token.clone();
        let events = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(async {
                let provider = provider.clone();
                let task = tokio::spawn(async move {
                    provider
                        .stream(
                            ModelRequest {
                                messages: vec![Message::text(Role::User, "hello")],
                                tools: Vec::new(),
                            },
                            &token,
                        )
                        .await
                        .unwrap()
                        .try_collect::<Vec<_>>()
                        .await
                        .unwrap()
                });
                tokio::time::sleep(Duration::from_millis(25)).await;
                trigger.cancel();
                task.await.unwrap()
            });
        server.join().unwrap();
        assert!(matches!(
            events.first(),
            Some(ModelEvent::RetryScheduled { .. })
        ));
        assert!(matches!(events.last(), Some(ModelEvent::Cancelled)));
    }

    #[test]
    fn openai_provider_preserves_an_explicit_versioned_base_url() {
        let response = json!({"choices":[{"message":{"role":"assistant","content":"ok"}}]});
        let (base_url, captured, server) = fake_http_server(200, response);
        let provider =
            OpenAiCompatibleProvider::new(OpenAiConfig::new(base_url, "gpt-test")).unwrap();

        collect_response(
            &provider,
            ModelRequest {
                messages: vec![Message::text(Role::User, "hello")],
                tools: Vec::new(),
            },
        )
        .unwrap();
        let request = captured.recv().unwrap();
        server.join().unwrap();

        assert!(request.starts_with("POST /v1/chat/completions HTTP/1.1"));
    }

    #[test]
    fn openai_provider_reports_bounded_http_diagnostics_for_html_success() {
        let (base_url, _captured, server) = fake_http_server_raw(
            200,
            "text/html; charset=utf-8",
            "<html><title>Focus proxy</title></html>".into(),
        );
        let provider = OpenAiCompatibleProvider::new(
            OpenAiConfig::new(base_url, "gpt-test").with_api_key("secret-token"),
        )
        .unwrap();

        let error = collect_response(
            &provider,
            ModelRequest {
                messages: vec![Message::text(Role::User, "hello")],
                tools: Vec::new(),
            },
        )
        .unwrap_err()
        .to_string();
        server.join().unwrap();

        assert!(error.contains("status=200"), "{error}");
        assert!(error.contains("content-type=text/html"), "{error}");
        assert!(error.contains("Focus proxy"), "{error}");
        assert!(!error.contains("secret-token"), "{error}");
        assert!(error.len() < 2_000, "{error}");
    }

    #[test]
    fn openai_provider_reports_a_transport_category_for_send_failures() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        drop(listener);
        let provider = OpenAiCompatibleProvider::new(
            OpenAiConfig::new(format!("http://{address}/v1"), "gpt-test").with_max_retries(0),
        )
        .unwrap();

        let error = collect_response(
            &provider,
            ModelRequest {
                messages: vec![Message::text(Role::User, "hello")],
                tools: Vec::new(),
            },
        )
        .unwrap_err()
        .to_string();

        assert!(error.contains("transport category=connect"), "{error}");
    }

    #[test]
    fn default_openai_transport_backoff_is_bounded_for_transient_routes() {
        assert_eq!(
            retry_delay(DEFAULT_OPENAI_RETRY_BASE_DELAY, 0),
            Duration::from_millis(250)
        );
        assert_eq!(
            retry_delay(DEFAULT_OPENAI_RETRY_BASE_DELAY, 1),
            Duration::from_millis(500)
        );
    }

    #[test]
    fn openai_provider_bounds_and_redacts_http_errors() {
        let (base_url, _captured, server) =
            fake_http_server(401, json!({"error":{"message":"secret-token denied"}}));
        let provider = OpenAiCompatibleProvider::new(
            OpenAiConfig::new(base_url, "gpt-test").with_api_key("secret-token"),
        )
        .unwrap();

        let error = collect_response(
            &provider,
            ModelRequest {
                messages: vec![Message::text(Role::User, "hello")],
                tools: Vec::new(),
            },
        )
        .unwrap_err()
        .to_string();
        server.join().unwrap();

        assert!(error.contains("401"));
        assert!(!error.contains("secret-token"));
        assert!(error.len() < 10_000);
    }

    #[test]
    fn openai_provider_debug_never_exposes_the_api_key() {
        let provider = OpenAiCompatibleProvider::new(
            OpenAiConfig::new("http://127.0.0.1:1/v1", "gpt-test").with_api_key("debug-secret"),
        )
        .unwrap();

        let debug = format!("{provider:?}");

        assert!(!debug.contains("debug-secret"));
        assert!(debug.contains("api_key_configured"));
    }

    #[test]
    fn openai_provider_cancels_an_in_flight_http_request() {
        let (base_url, request_received, server) = stalling_http_server();
        let provider = OpenAiCompatibleProvider::new(
            OpenAiConfig::new(base_url, "gpt-test").with_timeout(Duration::from_secs(2)),
        )
        .unwrap();
        let cancellation = CancellationToken::default();
        let worker_cancellation = cancellation.clone();
        let worker = thread::spawn(move || {
            use futures::TryStreamExt;

            tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap()
                .block_on(async {
                    provider
                        .stream(
                            ModelRequest {
                                messages: vec![Message::text(Role::User, "hello")],
                                tools: Vec::new(),
                            },
                            &worker_cancellation,
                        )
                        .await?
                        .try_collect::<Vec<_>>()
                        .await
                })
        });
        request_received
            .recv_timeout(Duration::from_secs(1))
            .unwrap();

        let started = Instant::now();
        cancellation.cancel();
        let result = worker.join().unwrap();
        server.join().unwrap();

        assert!(
            matches!(result, Ok(events) if matches!(events.as_slice(), [ModelEvent::Cancelled]))
        );
        assert!(started.elapsed() < Duration::from_secs(1));
    }

    #[cfg(windows)]
    #[test]
    fn command_provider_kills_and_reaps_the_child_when_cancelled() {
        let _process_guard = crate::sandbox::process_test_lock().lock().unwrap();
        let pid_file =
            std::env::temp_dir().join(format!("pi-command-provider-{}.pid", uuid::Uuid::new_v4()));
        let provider = CommandModelProvider::new(
            std::env::current_exe().unwrap(),
            vec![
                "--exact".into(),
                "provider::tests::command_provider_cancellation_helper".into(),
                "--ignored".into(),
                "--nocapture".into(),
                "--".into(),
                pid_file.display().to_string(),
            ],
        );
        let cancellation = CancellationToken::default();
        let worker_cancellation = cancellation.clone();
        let worker = thread::spawn(move || {
            provider.complete_request(
                ModelRequest {
                    messages: vec![Message::text(Role::User, "hello")],
                    tools: Vec::new(),
                },
                &worker_cancellation,
            )
        });
        let deadline = Instant::now() + Duration::from_secs(5);
        let pid = loop {
            match std::fs::read_to_string(&pid_file) {
                Ok(value) if !value.trim().is_empty() => break value.trim().to_owned(),
                Ok(_) | Err(_) if Instant::now() < deadline => {
                    thread::sleep(Duration::from_millis(10));
                }
                Err(error) => panic!("command adapter did not publish a readable PID: {error}"),
                Ok(_) => panic!("command adapter published an empty PID"),
            }
        };

        let started = Instant::now();
        cancellation.cancel();
        let result = worker.join().unwrap();
        let process_id = pid.parse().expect("command adapter PID was not numeric");
        let _ = std::fs::remove_file(pid_file);

        assert!(matches!(result, Err(KernelError::Cancelled)));
        assert!(started.elapsed() < Duration::from_secs(5));
        assert!(wait_for_process_exit(process_id, Duration::from_secs(1)));
    }

    #[test]
    #[ignore = "subprocess helper for command provider process-tree cancellation coverage"]
    fn command_provider_cancellation_helper() {
        let pid_file = std::env::args().next_back().unwrap();
        std::fs::write(pid_file, std::process::id().to_string()).unwrap();
        thread::sleep(Duration::from_secs(10));
    }

    fn fake_http_server(
        status: u16,
        response: Value,
    ) -> (String, mpsc::Receiver<String>, thread::JoinHandle<()>) {
        fake_http_server_raw(status, "application/json", response.to_string())
    }

    fn fake_http_server_raw(
        status: u16,
        content_type: &str,
        body: String,
    ) -> (String, mpsc::Receiver<String>, thread::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let (sender, receiver) = mpsc::channel();
        let content_type = content_type.to_owned();
        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut buffer = Vec::new();
            let mut chunk = [0_u8; 4096];
            let header_end;
            loop {
                let read = stream.read(&mut chunk).unwrap();
                assert!(read > 0);
                buffer.extend_from_slice(&chunk[..read]);
                if let Some(index) = find_bytes(&buffer, b"\r\n\r\n") {
                    header_end = index + 4;
                    break;
                }
            }
            let headers = String::from_utf8_lossy(&buffer[..header_end]);
            let content_length = headers
                .lines()
                .find_map(|line| {
                    let (name, value) = line.split_once(':')?;
                    name.eq_ignore_ascii_case("content-length")
                        .then(|| value.trim().parse::<usize>().unwrap())
                })
                .unwrap_or(0);
            while buffer.len() < header_end + content_length {
                let read = stream.read(&mut chunk).unwrap();
                assert!(read > 0);
                buffer.extend_from_slice(&chunk[..read]);
            }
            sender
                .send(String::from_utf8_lossy(&buffer).into_owned())
                .unwrap();
            let reason = if status == 200 { "OK" } else { "Error" };
            write!(
                stream,
                "HTTP/1.1 {status} {reason}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            )
            .unwrap();
        });
        (format!("http://{address}/v1"), receiver, server)
    }

    fn multi_response_http_server(
        responses: Vec<(u16, String)>,
    ) -> (String, mpsc::Receiver<String>, thread::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let (sender, receiver) = mpsc::channel();
        let server = thread::spawn(move || {
            for (status, body) in responses {
                let (mut stream, _) = listener.accept().unwrap();
                let request = read_http_request(&mut stream);
                sender.send(request).unwrap();
                let reason = if status == 200 { "OK" } else { "Error" };
                let _ = write!(
                    stream,
                    "HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                );
            }
        });
        (format!("http://{address}/v1"), receiver, server)
    }

    fn read_http_request(stream: &mut std::net::TcpStream) -> String {
        let mut buffer = Vec::new();
        let mut chunk = [0_u8; 4096];
        let header_end;
        loop {
            let read = stream.read(&mut chunk).unwrap();
            assert!(read > 0);
            buffer.extend_from_slice(&chunk[..read]);
            if let Some(index) = find_bytes(&buffer, b"\r\n\r\n") {
                header_end = index + 4;
                break;
            }
        }
        let headers = String::from_utf8_lossy(&buffer[..header_end]);
        let content_length = headers
            .lines()
            .find_map(|line| {
                let (name, value) = line.split_once(':')?;
                name.eq_ignore_ascii_case("content-length")
                    .then(|| value.trim().parse::<usize>().unwrap())
            })
            .unwrap_or(0);
        while buffer.len() < header_end + content_length {
            let read = stream.read(&mut chunk).unwrap();
            assert!(read > 0);
            buffer.extend_from_slice(&chunk[..read]);
        }
        String::from_utf8_lossy(&buffer).into_owned()
    }

    fn sse_http_server(
        frames: Vec<String>,
        inter_frame_delay: Duration,
    ) -> (String, mpsc::Receiver<String>, thread::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let (sender, receiver) = mpsc::channel();
        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut buffer = Vec::new();
            let mut chunk = [0_u8; 4096];
            let header_end;
            loop {
                let read = stream.read(&mut chunk).unwrap();
                assert!(read > 0);
                buffer.extend_from_slice(&chunk[..read]);
                if let Some(index) = find_bytes(&buffer, b"\r\n\r\n") {
                    header_end = index + 4;
                    break;
                }
            }
            let headers = String::from_utf8_lossy(&buffer[..header_end]);
            let content_length = headers
                .lines()
                .find_map(|line| {
                    let (name, value) = line.split_once(':')?;
                    name.eq_ignore_ascii_case("content-length")
                        .then(|| value.trim().parse::<usize>().unwrap())
                })
                .unwrap_or(0);
            while buffer.len() < header_end + content_length {
                let read = stream.read(&mut chunk).unwrap();
                assert!(read > 0);
                buffer.extend_from_slice(&chunk[..read]);
            }
            sender
                .send(String::from_utf8_lossy(&buffer).into_owned())
                .unwrap();
            let body_len = frames.iter().map(String::len).sum::<usize>();
            write!(
                stream,
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {body_len}\r\nConnection: close\r\n\r\n"
            )
            .unwrap();
            stream.flush().unwrap();
            for frame in frames {
                if stream.write_all(frame.as_bytes()).is_err() || stream.flush().is_err() {
                    return;
                }
                thread::sleep(inter_frame_delay);
            }
        });
        (format!("http://{address}/v1"), receiver, server)
    }

    type ResponseSseFixture = (Vec<(String, String)>, Vec<String>);

    fn response_sse_server(
        responses: Vec<ResponseSseFixture>,
    ) -> (String, mpsc::Receiver<String>, thread::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let (sender, receiver) = mpsc::channel();
        let server = thread::spawn(move || {
            for (extra_headers, frames) in responses {
                let (mut stream, _) = listener.accept().unwrap();
                let request = read_http_request(&mut stream);
                sender.send(request).unwrap();
                let body = frames.concat();
                write!(
                    stream,
                    "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n",
                    body.len()
                )
                .unwrap();
                for (name, value) in extra_headers {
                    write!(stream, "{name}: {value}\r\n").unwrap();
                }
                write!(stream, "\r\n{body}").unwrap();
                stream.flush().unwrap();
            }
        });
        (format!("http://{address}/v1"), receiver, server)
    }

    fn stalling_http_server() -> (String, mpsc::Receiver<()>, thread::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let (sender, receiver) = mpsc::channel();
        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut buffer = Vec::new();
            let mut chunk = [0_u8; 4096];
            let header_end;
            loop {
                let read = stream.read(&mut chunk).unwrap();
                assert!(read > 0);
                buffer.extend_from_slice(&chunk[..read]);
                if let Some(index) = find_bytes(&buffer, b"\r\n\r\n") {
                    header_end = index + 4;
                    break;
                }
            }
            let headers = String::from_utf8_lossy(&buffer[..header_end]);
            let content_length = headers
                .lines()
                .find_map(|line| {
                    let (name, value) = line.split_once(':')?;
                    name.eq_ignore_ascii_case("content-length")
                        .then(|| value.trim().parse::<usize>().unwrap())
                })
                .unwrap_or(0);
            while buffer.len() < header_end + content_length {
                let read = stream.read(&mut chunk).unwrap();
                assert!(read > 0);
                buffer.extend_from_slice(&chunk[..read]);
            }
            sender.send(()).unwrap();
            stream
                .set_read_timeout(Some(Duration::from_secs(3)))
                .unwrap();
            let _ = stream.read(&mut [0_u8; 1]);
        });
        (format!("http://{address}/v1"), receiver, server)
    }

    fn split_http_request(request: &str) -> (&str, &str) {
        request.split_once("\r\n\r\n").unwrap()
    }

    fn find_bytes(haystack: &[u8], needle: &[u8]) -> Option<usize> {
        haystack
            .windows(needle.len())
            .position(|window| window == needle)
    }

    #[test]
    fn openai_sse_normalizes_provider_reasoning_summary_deltas() {
        let mut decoder = OpenAiSseDecoder::new(
            "fixture".into(),
            "gpt-test".into(),
            "https://example.test/v1/chat/completions".into(),
            None,
            4_096,
        );

        decoder
            .consume_frame(
                br#"data: {"choices":[{"delta":{"reasoning_content":"**Inspect** the workspace"}}]}

"#,
            )
            .unwrap();

        assert!(decoder.pending.iter().any(|event| matches!(
            event,
            ModelEvent::ReasoningSummaryDelta { text } if text == "**Inspect** the workspace"
        )));
    }
}
