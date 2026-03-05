export declare const joinBuffers: (buffers: Uint8Array[]) => Uint8Array;
/**
 * Convert list of bytes (number) to text
 * @param buffer
 * @returns a string
 */
export declare const bufToText: (buffer: ArrayBuffer | Uint8Array) => string;
/**
 * Get default stdout/stderr config for wasm module
 */
export declare const getWModuleConfig: (pathConfig: {
    [filename: string]: string;
}) => {
    noInitialRun: boolean;
    print: (text: any) => void;
    printErr: (text: any) => void;
    locateFile: (filename: string, basePath: string) => string;
};
export interface ShardInfo {
    baseURL: string;
    current: number;
    total: number;
}
/**
 * Parse shard number and total from a file name or URL
 */
export declare const parseShardNumber: (fnameOrUrl: string) => ShardInfo;
/**
 * Parses a model URL and returns an array of URLs based on the following patterns:
 * - If the input URL is an array, it returns the array itself.
 * - If the input URL is a string in the `gguf-split` format, it returns an array containing the URL of each shard in ascending order.
 * - Otherwise, it returns an array containing the input URL as a single element array.
 * @param modelUrl URL or list of URLs
 */
export declare const parseModelUrl: (modelUrl: string) => string[];
/**
 * Check if the given blobs are files or not, then sort them by shard number
 */
export declare const sortFileByShard: (blobs: Blob[]) => void;
export declare const delay: (ms: number) => Promise<unknown>;
export declare const absoluteUrl: (relativePath: string) => string;
export declare const padDigits: (number: number, digits: number) => string;
export declare const sumArr: (arr: number[]) => number;
export declare const isString: (value: any) => boolean;
/**
 * Browser feature detection
 * Copied from https://unpkg.com/wasm-feature-detect?module (Apache License)
 */
/**
 * @returns true if browser support multi-threads
 */
export declare const isSupportMultiThread: () => Promise<boolean>;
/**
 * Throws an error if the environment is not compatible
 */
export declare const checkEnvironmentCompatible: () => Promise<void>;
/**
 * Check if browser is Safari
 * Source: https://github.com/DamonOehlman/detect-browser/blob/master/src/index.ts
 */
export declare const isSafari: () => boolean;
/**
 * Check if browser is Safari iOS / iPad / iPhone
 * Source: https://github.com/DamonOehlman/detect-browser/blob/master/src/index.ts
 */
export declare const isSafariMobile: () => boolean;
/**
 * Create a worker from a string
 */
export declare const createWorker: (workerCode: string | Blob) => Worker;
