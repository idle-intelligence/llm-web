import { ProxyToWorker } from './worker.js';
import { absoluteUrl, bufToText, checkEnvironmentCompatible, isString, isSupportMultiThread, joinBuffers, sortFileByShard, padDigits, } from './utils.js';
import CacheManager from './cache-manager.js';
import { ModelManager, Model } from './model-manager.js';
const HF_MODEL_ID_REGEX = /^([a-zA-Z0-9_\-\.]+)\/([a-zA-Z0-9_\-\.]+)$/;
const HF_MODEL_ID_REGEX_EXPLAIN = "Hugging Face model ID is incorrect. Only regular alphanumeric characters, '-', '.' and '_' supported";
/**
 * Logger preset with debug messages suppressed
 */
export const LoggerWithoutDebug = {
    ...console,
    debug: () => { },
};
export class WllamaError extends Error {
    type;
    constructor(message, type = 'unknown_error') {
        super(message);
        this.type = type;
    }
}
export class Wllama {
    // The CacheManager and ModelManager are singleton, can be accessed by user
    cacheManager;
    modelManager;
    proxy = null;
    config;
    pathConfig;
    useMultiThread = false;
    useEmbeddings = false;
    // available when loaded
    loadedContextInfo = null;
    bosToken = -1;
    eosToken = -1;
    eotToken = -1;
    addBosToken = false;
    addEosToken = false;
    chatTemplate;
    metadata;
    samplingConfig = {};
    hasEncoder = false;
    decoderStartToken = -1;
    nCachedTokens = 0;
    constructor(pathConfig, wllamaConfig = {}) {
        checkEnvironmentCompatible();
        if (!pathConfig)
            throw new WllamaError('AssetsPathConfig is required');
        this.pathConfig = pathConfig;
        this.config = wllamaConfig;
        this.cacheManager = wllamaConfig.cacheManager ?? new CacheManager();
        this.modelManager =
            wllamaConfig.modelManager ??
                new ModelManager({
                    cacheManager: this.cacheManager,
                    logger: wllamaConfig.logger ?? console,
                    parallelDownloads: wllamaConfig.parallelDownloads,
                    allowOffline: wllamaConfig.allowOffline,
                });
    }
    logger() {
        return this.config.logger ?? console;
    }
    checkModelLoaded() {
        if (!this.isModelLoaded()) {
            throw new WllamaError('loadModel() is not yet called', 'model_not_loaded');
        }
    }
    /**
     * Check if the model is loaded via `loadModel()`
     */
    isModelLoaded() {
        return !!this.proxy && !!this.metadata;
    }
    /**
     * Get token ID associated to BOS (begin of sentence) token.
     *
     * NOTE: This can only being used after `loadModel` is called.
     *
     * @returns -1 if the model is not loaded.
     */
    getBOS() {
        return this.bosToken;
    }
    /**
     * Get token ID associated to EOS (end of sentence) token.
     *
     * NOTE: This can only being used after `loadModel` is called.
     *
     * @returns -1 if the model is not loaded.
     */
    getEOS() {
        return this.eosToken;
    }
    /**
     * Get token ID associated to EOT (end of turn) token.
     *
     * NOTE: This can only being used after `loadModel` is called.
     *
     * @returns -1 if the model is not loaded.
     */
    getEOT() {
        return this.eotToken;
    }
    /**
     * Get token ID associated to token used by decoder, to start generating output sequence(only usable for encoder-decoder architecture). In other words, encoder uses normal BOS and decoder uses this token.
     *
     * NOTE: This can only being used after `loadModel` is called.
     *
     * @returns -1 if the model is not loaded.
     */
    getDecoderStartToken() {
        return this.decoderStartToken;
    }
    /**
     * Get model hyper-parameters and metadata
     *
     * NOTE: This can only being used after `loadModel` is called.
     *
     * @returns ModelMetadata
     */
    getModelMetadata() {
        this.checkModelLoaded();
        return this.metadata;
    }
    /**
     * Check if we're currently using multi-thread build.
     *
     * NOTE: This can only being used after `loadModel` is called.
     *
     * @returns true if multi-thread is used.
     */
    isMultithread() {
        this.checkModelLoaded();
        return this.useMultiThread;
    }
    /**
     * Check if the current model uses encoder-decoder architecture
     *
     * NOTE: This can only being used after `loadModel` is called.
     *
     * @returns true if multi-thread is used.
     */
    isEncoderDecoderArchitecture() {
        this.checkModelLoaded();
        return this.hasEncoder;
    }
    /**
     * Must we add BOS token to the tokenized sequence?
     *
     * NOTE: This can only being used after `loadModel` is called.
     *
     * @returns true if BOS token must be added to the sequence
     */
    mustAddBosToken() {
        this.checkModelLoaded();
        return this.addBosToken;
    }
    /**
     * Must we add EOS token to the tokenized sequence?
     *
     * NOTE: This can only being used after `loadModel` is called.
     *
     * @returns true if EOS token must be added to the sequence
     */
    mustAddEosToken() {
        this.checkModelLoaded();
        return this.addEosToken;
    }
    /**
     * Get the jinja chat template comes with the model. It only available if the original model (before converting to gguf) has the template in `tokenizer_config.json`
     *
     * NOTE: This can only being used after `loadModel` is called.
     *
     * @returns the jinja template. null if there is no template in gguf
     */
    getChatTemplate() {
        this.checkModelLoaded();
        return this.chatTemplate ?? null;
    }
    /**
     * Load model from a given URL (or a list of URLs, in case the model is splitted into smaller files)
     * - If the model already been downloaded (via `downloadModel()`), then we will use the cached model
     * - Else, we download the model from internet
     * @param modelUrl URL to the GGUF file. If the model is splitted, pass the URL to the first shard.
     * @param config
     */
    async loadModelFromUrl(modelUrl, config = {}) {
        const url = isString(modelUrl) ? modelUrl : modelUrl[0];
        const useCache = config.useCache ?? true;
        const model = useCache
            ? await this.modelManager.getModelOrDownload(url, config)
            : await this.modelManager.downloadModel(url, config);
        const blobs = await model.open();
        return await this.loadModel(blobs, config);
    }
    /**
     * Load model from a given Hugging Face model ID and file path.
     *
     * @param modelId The HF model ID, for example: 'ggml-org/models'
     * @param filePath The GGUF file path, for example: 'tinyllamas/stories15M-q4_0.gguf'
     * @param config
     */
    async loadModelFromHF(modelId, filePath, config = {}) {
        if (!modelId.match(HF_MODEL_ID_REGEX)) {
            throw new WllamaError(HF_MODEL_ID_REGEX_EXPLAIN, 'download_error');
        }
        if (!filePath.endsWith('.gguf')) {
            throw new WllamaError('Only GGUF file is supported', 'download_error');
        }
        return await this.loadModelFromUrl(`https://huggingface.co/${modelId}/resolve/main/${filePath}`, config);
    }
    /**
     * Load model from a given list of Blob.
     *
     * You can pass multiple buffers into the function (in case the model contains multiple shards).
     *
     * @param ggufBlobsOrModel Can be either list of Blobs (in case you use local file), or a Model object (in case you use ModelManager)
     * @param config LoadModelConfig
     */
    async loadModel(ggufBlobsOrModel, config = {}) {
        const blobs = ggufBlobsOrModel instanceof Model
            ? await ggufBlobsOrModel.open()
            : [...ggufBlobsOrModel]; // copy array
        if (blobs.some((b) => b.size === 0)) {
            throw new WllamaError('Input model (or splits) must be non-empty Blob or File', 'load_error');
        }
        sortFileByShard(blobs);
        const hasMultipleBuffers = blobs.length > 1;
        if (this.proxy) {
            throw new WllamaError('Module is already initialized', 'load_error');
        }
        // detect if we can use multi-thread
        const supportMultiThread = await isSupportMultiThread();
        if (!supportMultiThread) {
            this.logger().warn('Multi-threads are not supported in this environment, falling back to single-thread');
        }
        const hasPathMultiThread = !!this.pathConfig['multi-thread/wllama.wasm'];
        if (!hasPathMultiThread) {
            this.logger().warn('Missing paths to "multi-thread/wllama.wasm", falling back to single-thread');
        }
        const hwConccurency = Math.floor((navigator.hardwareConcurrency || 1) / 2);
        const nbThreads = config.n_threads ?? hwConccurency;
        this.useMultiThread =
            supportMultiThread && hasPathMultiThread && nbThreads > 1;
        const mPathConfig = this.useMultiThread
            ? {
                'wllama.wasm': absoluteUrl(this.pathConfig['multi-thread/wllama.wasm']),
            }
            : {
                'wllama.wasm': absoluteUrl(this.pathConfig['single-thread/wllama.wasm']),
            };
        this.proxy = new ProxyToWorker(mPathConfig, this.useMultiThread ? nbThreads : 1, this.config.suppressNativeLog ?? false, this.logger());
        // TODO: files maybe out-of-order
        await this.proxy.moduleInit(blobs.map((blob, i) => ({
            name: hasMultipleBuffers
                ? `model-${padDigits(i + 1, 5)}-of-${padDigits(blobs.length, 5)}.gguf`
                : 'model.gguf',
            blob,
        })));
        // run it
        const startResult = await this.proxy.wllamaStart();
        if (!startResult.success) {
            throw new WllamaError(`Error while calling start function, result = ${startResult}`);
        }
        // load the model
        const loadResult = await this.proxy.wllamaAction('load', {
            ...config,
            use_mmap: true,
            use_mlock: true,
            seed: config.seed || Math.floor(Math.random() * 100000),
            n_ctx: config.n_ctx || 1024,
            n_threads: this.useMultiThread ? nbThreads : 1,
            model_path: hasMultipleBuffers
                ? `/models/model-00001-of-${padDigits(blobs.length, 5)}.gguf`
                : '/models/model.gguf',
        });
        this.bosToken = loadResult.token_bos;
        this.eosToken = loadResult.token_eos;
        this.eotToken = loadResult.token_eot;
        this.useEmbeddings = !!config.embeddings;
        this.metadata = {
            hparams: {
                nVocab: loadResult.n_vocab,
                nCtxTrain: loadResult.n_ctx_train,
                nEmbd: loadResult.n_embd,
                nLayer: loadResult.n_layer,
            },
            meta: loadResult.metadata,
        };
        this.hasEncoder = !!loadResult.has_encoder;
        this.decoderStartToken = loadResult.token_decoder_start;
        this.addBosToken = loadResult.add_bos_token;
        this.addEosToken = loadResult.add_eos_token;
        this.chatTemplate = loadResult.metadata['tokenizer.chat_template'];
        this.loadedContextInfo = loadResult;
        this.logger().debug({ loadResult });
    }
    getLoadedContextInfo() {
        this.checkModelLoaded();
        if (!this.loadedContextInfo) {
            throw new WllamaError('Loaded context info is not available');
        }
        // copy object
        return { ...this.loadedContextInfo };
    }
    //////////////////////////////////////////////
    // High level API
    /**
     * Calculate embedding vector for a given text.
     * By default, BOS and EOS tokens will be added automatically. You can use the "skipBOS" and "skipEOS" option to disable it.
     * @param text Input text
     * @returns An embedding vector
     */
    async createEmbedding(text, options = {}) {
        this.checkModelLoaded();
        const opt = {
            skipBOS: false,
            skipEOS: false,
            ...options,
        };
        await this.samplingInit(this.samplingConfig);
        await this.kvClear();
        const tokens = await this.tokenize(text);
        if (this.bosToken && !opt.skipBOS) {
            tokens.unshift(this.bosToken);
        }
        if (this.eosToken && !opt.skipEOS) {
            tokens.push(this.eosToken);
        }
        const result = await this.embeddings(tokens);
        return result;
    }
    /**
     * Make completion for a given chat messages.
     *
     * NOTE: this function uses the chat template (if available) to format the chat messages. If the template is not available, it will use the default format (chatml). It can throw an error if the chat template is not compatible.
     *
     * @param messages Chat messages
     * @param options
     * @returns Output completion text (only the completion part)
     */
    async createChatCompletion(messages, options) {
        const prompt = await this.formatChat(messages, true);
        return await this.createCompletion(prompt, options);
    }
    /**
     * Make completion for a given text.
     * @param prompt Input text
     * @param options
     * @returns Output completion text (only the completion part)
     */
    async createCompletion(prompt, options) {
        this.checkModelLoaded();
        this.samplingConfig = options.sampling ?? {};
        await this.samplingInit(this.samplingConfig);
        const stopTokens = [
            this.eosToken,
            this.eotToken,
            ...(options.stopTokens ?? []),
        ];
        // process prompt
        let tokens = await this.tokenize(prompt, true);
        if (this.addBosToken && tokens[0] !== this.bosToken) {
            tokens.unshift(this.bosToken);
        }
        // maybe reuse KV cache
        if (options.useCache) {
            tokens = await this.computeNonCachedTokens(tokens);
        }
        else {
            await this.kvClear();
        }
        // decode/encode tokens
        await this.samplingAccept(tokens);
        if (this.isEncoderDecoderArchitecture()) {
            await this.encode(tokens);
            await this.decode([this.getDecoderStartToken()], {});
        }
        else {
            await this.decode(tokens, {});
        }
        let outBuf = new Uint8Array();
        // abort signal
        let abort = false;
        const abortSignal = () => {
            abort = true;
        };
        // predict next tokens
        for (let i = 0; i < (options.nPredict ?? Infinity); i++) {
            const sampled = await this.samplingSample();
            if (stopTokens.includes(sampled.token)) {
                break; // stop token
            }
            // @ts-ignore Type 'Uint8Array<ArrayBufferLike>' is not assignable to type 'Uint8Array<ArrayBuffer>'
            outBuf = joinBuffers([outBuf, sampled.piece]);
            if (options.onNewToken) {
                options.onNewToken(sampled.token, sampled.piece, bufToText(outBuf), {
                    abortSignal,
                });
            }
            if (abort) {
                break; // abort signal is set
            }
            // decode next token
            await this.samplingAccept([sampled.token]);
            await this.decode([sampled.token], {});
        }
        return bufToText(outBuf);
    }
    //////////////////////////////////////////////
    // Low level API
    /**
     * Create or reset the ctx_sampling
     * @param config
     * @param pastTokens In case re-initializing the ctx_sampling, you can re-import past tokens into the new context
     */
    async samplingInit(config, pastTokens = []) {
        this.checkModelLoaded();
        this.samplingConfig = config;
        const result = await this.proxy.wllamaAction('sampling_init', {
            ...config,
            tokens: pastTokens,
        });
        if (!result.success) {
            throw new WllamaError('Failed to initialize sampling');
        }
    }
    /**
     * Get a list of pieces in vocab.
     * NOTE: This function is slow, should only be used once.
     * @returns A list of Uint8Array. The nth element in the list associated to nth token in vocab
     */
    async getVocab() {
        this.checkModelLoaded();
        const result = await this.proxy.wllamaAction('get_vocab', {});
        return result.vocab.map((arr) => new Uint8Array(arr));
    }
    /**
     * Lookup to see if a token exist in vocab or not. Useful for searching special tokens like "<|im_start|>"
     * NOTE: It will match the whole token, so do not use it as a replacement for tokenize()
     * @param piece
     * @returns Token ID associated to the given piece. Returns -1 if cannot find the token.
     */
    async lookupToken(piece) {
        this.checkModelLoaded();
        const result = await this.proxy.wllamaAction('lookup_token', { piece });
        if (!result.success) {
            return -1;
        }
        else {
            return result.token;
        }
    }
    /**
     * Convert a given text to list of tokens
     * @param text
     * @param special Should split special tokens?
     * @returns List of token ID
     */
    async tokenize(text, special = true) {
        this.checkModelLoaded();
        const result = await this.proxy.wllamaAction('tokenize', special ? { text, special: true } : { text });
        return result.tokens;
    }
    /**
     * Convert a list of tokens to text
     * @param tokens
     * @returns Uint8Array, which maybe an unfinished unicode
     */
    async detokenize(tokens) {
        this.checkModelLoaded();
        const result = await this.proxy.wllamaAction('detokenize', { tokens });
        return new Uint8Array(result.buffer);
    }
    /**
     * Run llama_decode()
     * @param tokens A list of tokens to be decoded
     * @param options
     * @returns n_past (number of tokens so far in the sequence)
     */
    async decode(tokens, options) {
        this.checkModelLoaded();
        if (this.useEmbeddings) {
            throw new WllamaError('embeddings is enabled. Use wllama.setOptions({ embeddings: false }) to disable it.');
        }
        if (tokens.length === 0) {
            // do not call llama_decode if list of tokens is empty
            return {
                nPast: this.nCachedTokens,
            };
        }
        if (this.nCachedTokens + tokens.length > this.loadedContextInfo.n_ctx) {
            throw new WllamaError('Running out of context cache. Please increase n_ctx when loading the model', 'kv_cache_full');
        }
        const batches = this.breakTokensIntoBatches(tokens, this.loadedContextInfo.n_batch);
        let result;
        for (let i = 0; i < batches.length; i++) {
            const isNotLast = batches.length > 1 && i < batches.length - 1;
            result = await this.proxy.wllamaAction('decode', {
                tokens: batches[i],
                skip_logits: options.skipLogits || isNotLast,
            });
            if (result.error) {
                throw new WllamaError(result.error);
            }
            else if (!result.success) {
                throw new WllamaError('Cannot encode, unknown error');
            }
        }
        this.nCachedTokens = result.n_past;
        return { nPast: result.n_past };
    }
    /**
     * Run llama_encode()
     * @param tokens A list of tokens to be encoded
     * @param options Unused for now
     * @returns n_past (number of tokens so far in the sequence)
     */
    async encode(tokens, 
    // @ts-ignore unused variable
    options) {
        this.checkModelLoaded();
        if (!this.hasEncoder) {
            throw new WllamaError('This model does not use encoder-decoder architecture.', 'inference_error');
        }
        if (this.useEmbeddings) {
            throw new WllamaError('embeddings is enabled. Use wllama.setOptions({ embeddings: false }) to disable it.', 'inference_error');
        }
        if (tokens.length === 0) {
            // do not call llama_encode if list of tokens is empty
            return {
                nPast: this.nCachedTokens,
            };
        }
        if (this.nCachedTokens + tokens.length > this.loadedContextInfo.n_ctx) {
            throw new WllamaError('Running out of context cache. Please increase n_ctx when loading the model', 'kv_cache_full');
        }
        const batches = this.breakTokensIntoBatches(tokens, this.loadedContextInfo.n_batch);
        let result;
        for (let i = 0; i < batches.length; i++) {
            result = await this.proxy.wllamaAction('encode', { tokens: batches[i] });
            if (result.error) {
                throw new WllamaError(result.error);
            }
            else if (!result.success) {
                throw new WllamaError('Cannot encode, unknown error');
            }
        }
        this.nCachedTokens = result.n_past;
        return { nPast: result.n_past };
    }
    breakTokensIntoBatches(tokens, maxBatchSize) {
        const batches = [];
        for (let i = 0; i < tokens.length; i += maxBatchSize) {
            batches.push(tokens.slice(i, i + maxBatchSize));
        }
        return batches;
    }
    /**
     * Sample a new token (remember to samplingInit() at least once before calling this function)
     * @returns the token ID and its detokenized value (which maybe an unfinished unicode)
     */
    async samplingSample() {
        this.checkModelLoaded();
        const result = await this.proxy.wllamaAction('sampling_sample', {});
        return {
            piece: new Uint8Array(result.piece),
            token: result.token,
        };
    }
    /**
     * Accept and save a new token to ctx_sampling
     * @param tokens
     */
    async samplingAccept(tokens) {
        this.checkModelLoaded();
        const result = await this.proxy.wllamaAction('sampling_accept', { tokens });
        if (!result.success) {
            throw new WllamaError('samplingAccept unknown error');
        }
    }
    /**
     * Get softmax-ed probability of logits, can be used for custom sampling
     * @param topK Get top K tokens having highest logits value. If topK == -1, we return all n_vocab logits, but this is not recommended because it's slow.
     */
    async getLogits(topK = 40) {
        this.checkModelLoaded();
        const result = await this.proxy.wllamaAction('get_logits', { top_k: topK });
        const logits = result.logits;
        return logits.map(([token, p]) => ({ token, p }));
    }
    /**
     * Calculate embeddings for a given list of tokens. Output vector is always normalized
     * @param tokens
     * @returns A list of number represents an embedding vector of N dimensions
     */
    async embeddings(tokens) {
        this.checkModelLoaded();
        if (!this.useEmbeddings) {
            throw new WllamaError('embeddings is disabled. Use wllama.setOptions({ embeddings: true }) to enable it.', 'inference_error');
        }
        if (this.nCachedTokens > 0) {
            this.logger().warn('Embeddings: KV cache is not empty, this may produce incorrect results');
        }
        if (this.nCachedTokens + tokens.length > this.loadedContextInfo.n_ctx) {
            throw new WllamaError('Running out of context cache. Please increase n_ctx when loading the model', 'kv_cache_full');
        }
        if (tokens.length > this.loadedContextInfo.n_batch) {
            throw new WllamaError('Embedding tokens does not fit into batch. Please increase n_batch when loading the model', 'inference_error');
        }
        if (tokens.length > this.loadedContextInfo.n_ubatch) {
            throw new WllamaError('Embedding tokens does not fit into physical batch. Please increase n_ubatch when loading the model', 'inference_error');
        }
        const result = await this.proxy.wllamaAction('embeddings', { tokens });
        if (result.error) {
            throw new WllamaError(result.error);
        }
        else if (!result.success) {
            throw new WllamaError('embeddings unknown error');
        }
        else {
            return result.embeddings;
        }
    }
    /**
     * Remove and shift some tokens from KV cache.
     * Keep n_keep, remove n_discard then shift the rest
     * @param nKeep
     * @param nDiscard
     */
    async kvRemove(nKeep, nDiscard) {
        this.checkModelLoaded();
        const result = await this.proxy.wllamaAction('kv_remove', {
            n_keep: nKeep,
            n_discard: nDiscard,
        });
        if (!result.success) {
            throw new WllamaError('kvRemove unknown error');
        }
        this.nCachedTokens -= nDiscard;
    }
    /**
     * Clear all tokens in KV cache
     */
    async kvClear() {
        this.checkModelLoaded();
        const result = await this.proxy.wllamaAction('kv_clear', {});
        if (!result.success) {
            throw new WllamaError('kvClear unknown error');
        }
        this.nCachedTokens = 0;
    }
    /**
     * Save session to file (virtual file system)
     * TODO: add ability to download the file
     * @param filePath
     * @returns List of tokens saved to the file
     */
    async sessionSave(filePath) {
        this.checkModelLoaded();
        const result = await this.proxy.wllamaAction('session_save', {
            session_path: filePath,
        });
        return result;
    }
    /**
     * Load session from file (virtual file system)
     * TODO: add ability to download the file
     * @param filePath
     */
    async sessionLoad(filePath) {
        this.checkModelLoaded();
        const result = await this.proxy.wllamaAction('session_load', {
            session_path: filePath,
        });
        if (result.error) {
            throw new WllamaError(result.error);
        }
        else if (!result.success) {
            throw new WllamaError('sessionLoad unknown error');
        }
        const cachedTokens = await this.getCachedTokens();
        this.nCachedTokens = cachedTokens.length;
    }
    /**
     * Apply chat template to a list of messages
     *
     * @param messages list of messages
     * @param addAssistant whether to add assistant prompt at the end
     * @param template (optional) custom template, see llama-server --chat-template argument for more details
     * @returns formatted chat
     */
    async formatChat(messages, addAssistant, template) {
        this.checkModelLoaded();
        const result = await this.proxy.wllamaAction('chat_format', {
            messages: messages,
            tmpl: template,
            add_ass: addAssistant,
        });
        if (result.error) {
            throw new WllamaError(result.error);
        }
        else if (!result.success) {
            throw new WllamaError('formatChat unknown error');
        }
        return result.formatted_chat;
    }
    /**
     * Set options for underlaying llama_context
     */
    async setOptions(opt) {
        this.checkModelLoaded();
        await this.proxy.wllamaAction('set_options', opt);
        this.useEmbeddings = opt.embeddings;
    }
    /**
     * Unload the model and free all memory.
     *
     * Note: This function will NOT crash if model is not yet loaded
     */
    async exit() {
        await this.proxy?.wllamaExit();
        this.proxy = null;
    }
    /**
     * get debug info
     */
    async _getDebugInfo() {
        this.checkModelLoaded();
        return await this.proxy.wllamaDebug();
    }
    ///// Prompt cache utils /////
    async getCachedTokens() {
        this.checkModelLoaded();
        const result = await this.proxy.wllamaAction('current_status', {});
        return result.tokens;
    }
    /**
     * Compare the input sequence and cachedToken, then return the part that is not in cache.
     * This function also remove mismatch part in cache (via kvRemove)
     */
    async computeNonCachedTokens(seq) {
        const cachedTokens = await this.getCachedTokens();
        let nKeep = 0;
        for (; nKeep < Math.min(cachedTokens.length, seq.length); nKeep++) {
            if (cachedTokens[nKeep] !== seq[nKeep]) {
                break;
            }
        }
        const nDiscard = cachedTokens.length - nKeep;
        this.logger().debug(`Cache nKeep=${nKeep} nDiscard=${nDiscard}`);
        if (nDiscard > 0) {
            await this.kvRemove(nKeep, nDiscard);
        }
        return seq.slice(nKeep, seq.length);
    }
}
//# sourceMappingURL=wllama.js.map