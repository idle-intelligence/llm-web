//! JavaScript bindings, bundled into lean's web package so a page does
//! `lean.generate(chatRender(template, messages, tools, true), ...)`.

use wasm_bindgen::prelude::*;

use crate::{oldest_exchange, render, Message};

fn from_js<T: serde::de::DeserializeOwned>(value: &JsValue, what: &str) -> Result<T, JsError> {
    let json = js_sys::JSON::stringify(value).map_err(|_| JsError::new(&format!("{what} is not JSON-serializable")))?;
    serde_json::from_str(&String::from(json)).map_err(|e| JsError::new(&format!("{what}: {e}")))
}

/// `render` for JavaScript: `template` is the `chat_template` string,
/// `messages` an array of `{role, content, tool_calls?, name?,
/// tool_call_id?}`, `tools` an array of tool definitions or null/undefined.
#[wasm_bindgen(js_name = chatRender)]
pub fn chat_render(template: &str, messages: JsValue, tools: JsValue, add_generation_prompt: bool) -> Result<String, JsError> {
    let messages: Vec<Message> = from_js(&messages, "messages")?;
    let tools: Option<Vec<serde_json::Value>> = if tools.is_undefined() || tools.is_null() { None } else { Some(from_js(&tools, "tools")?) };
    render(template, &messages, tools.as_deref(), add_generation_prompt).map_err(|e| JsError::new(&format!("{e:#}")))
}

/// `trim_oldest_exchange` for JavaScript: returns a new array without the
/// oldest exchange (the same message objects otherwise), or `undefined`
/// when there is no older exchange to drop. `messages` is not modified.
#[wasm_bindgen(js_name = chatTrimOldestExchange)]
pub fn chat_trim_oldest_exchange(messages: js_sys::Array) -> Result<JsValue, JsError> {
    let roles: Vec<String> = messages
        .iter()
        .map(|m| js_sys::Reflect::get(&m, &JsValue::from_str("role")).ok().and_then(|r| r.as_string()).ok_or_else(|| JsError::new("message missing 'role'")))
        .collect::<Result<_, _>>()?;
    let Some(range) = oldest_exchange(roles.iter().map(String::as_str)) else {
        return Ok(JsValue::UNDEFINED);
    };
    let kept = js_sys::Array::new();
    for (i, m) in messages.iter().enumerate() {
        if !range.contains(&i) {
            kept.push(&m);
        }
    }
    Ok(kept.into())
}
