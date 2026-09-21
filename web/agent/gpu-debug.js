/**
 * WebGPU validation diagnostics. Chrome's own console only prints cascade
 * errors once a pipeline/bind-group build fails ("[Invalid ...] is invalid
 * due to a previous error") — the root validation message is swallowed.
 * This module wraps the relevant GPUDevice methods in error scopes so the
 * real reason surfaces. Side-effect only: no behavior change, no exports
 * that affect control flow. Must be imported before pkg/llm_wasm.js runs
 * any GPU code (llm-worker.js imports this first).
 */

if (typeof GPUDevice !== 'undefined') {
  const shaderInfo = new WeakMap(); // GPUShaderModule -> {label, code}

  function summarizeEntries(entries) {
    if (!entries) return entries;
    return entries.map((e) => ({
      binding: e.binding,
      visibility: e.visibility,
      type: e.buffer?.type ?? e.type,
    }));
  }

  function summarizeBindGroupEntries(entries) {
    if (!entries) return entries;
    return entries.map((e) => ({
      binding: e.binding,
      size: e.resource?.size,
      offset: e.resource?.offset,
    }));
  }

  function describeDescriptor(method, descriptor) {
    if (!descriptor) return descriptor;
    try {
      switch (method) {
        case 'createShaderModule': {
          const lines = (descriptor.code || '').split('\n').slice(0, 40).join('\n');
          return { label: descriptor.label, code: lines };
        }
        case 'createBindGroupLayout':
          return { label: descriptor.label, entries: summarizeEntries(descriptor.entries) };
        case 'createPipelineLayout':
          return { label: descriptor.label, bindGroupLayouts: descriptor.bindGroupLayouts?.length };
        case 'createComputePipeline':
        case 'createComputePipelineAsync': {
          const module = descriptor.compute?.module;
          const info = module ? shaderInfo.get(module) : undefined;
          return {
            label: descriptor.label,
            entryPoint: descriptor.compute?.entryPoint,
            shaderLabel: info?.label,
            shaderCode: info?.code,
          };
        }
        case 'createBindGroup':
          return { label: descriptor.label, entries: summarizeBindGroupEntries(descriptor.entries) };
        default:
          return descriptor;
      }
    } catch (err) {
      return `<describe failed: ${err.message}>`;
    }
  }

  function wrap(proto, method) {
    const original = proto[method];
    if (typeof original !== 'function') return;
    proto[method] = function (...args) {
      this.pushErrorScope('validation');
      const result = original.apply(this, args);

      const finish = (res, err) => {
        if (err) {
          console.error(
            `[gpu-debug] ${method} failed:`,
            err.message,
            describeDescriptor(method, args[0])
          );
        } else if (method === 'createShaderModule' && res && typeof res.getCompilationInfo === 'function') {
          shaderInfo.set(res, { label: args[0]?.label, code: (args[0]?.code || '').split('\n').slice(0, 40).join('\n') });
          res.getCompilationInfo().then((info) => {
            if (info.messages.length) {
              console.error(
                '[gpu-debug] compilation:',
                info.messages.map((m) => `${m.type} ${m.lineNum}:${m.linePos} ${m.message}`).join('\n')
              );
            }
          });
        }
        return res;
      };

      if (result && typeof result.then === 'function') {
        return result.then((res) => this.popErrorScope().then((err) => finish(res, err))).then(
          (res) => res,
          (err) => {
            this.popErrorScope();
            throw err;
          }
        );
      }

      this.popErrorScope().then((err) => finish(result, err));
      return result;
    };
  }

  [
    'createShaderModule',
    'createBindGroupLayout',
    'createPipelineLayout',
    'createComputePipeline',
    'createBindGroup',
  ].forEach((m) => wrap(GPUDevice.prototype, m));

  // createComputePipelineAsync already returns a Promise<GPUComputePipeline>;
  // wrap() above handles both sync and async return shapes, but this method
  // resolves to the pipeline itself (not wrapped in the same error-scope
  // timing as the sync call), so wrap it explicitly too.
  if (typeof GPUDevice.prototype.createComputePipelineAsync === 'function') {
    wrap(GPUDevice.prototype, 'createComputePipelineAsync');
  }

  if (typeof GPUAdapter !== 'undefined' && typeof GPUAdapter.prototype.requestDevice === 'function') {
    const originalRequestDevice = GPUAdapter.prototype.requestDevice;
    GPUAdapter.prototype.requestDevice = function (descriptor) {
      console.error(
        '[gpu-debug] requestDevice:',
        JSON.stringify({
          requiredFeatures: descriptor?.requiredFeatures,
          requiredLimits: descriptor?.requiredLimits,
        })
      );
      return originalRequestDevice.call(this, descriptor).then((device) => {
        device.addEventListener('uncapturederror', (e) => {
          console.error('[gpu-debug] uncaptured:', e.error.message);
        });
        return device;
      });
    };
  }
}
