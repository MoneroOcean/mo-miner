declare module "core-js-pure/actual/json/parse.js" {
  const parse: typeof JSON.parse;
  export default parse;
}

declare function __momTimerStart(callback: () => unknown, delay: number, repeat: boolean): number;
declare function __momHandleClose(id: number): void;
declare function __momHandleRef(id: number, referenced: boolean): void;
declare function __momHandleHasRef(id: number): boolean;
declare function __momTCPConnect(host: string, port: number,
  callback: (event: string, value?: unknown) => void, timeout?: number | null): number;
declare function __momTCPWrite(id: number, data: Uint8Array): boolean;
declare function __momTCPEnd(id: number): void;
declare function __momGetEnv(name: string): string | undefined;
declare function __momSetEnv(name: string, value: string | null): void;
declare function __momEnvKeys(): string[];
declare function __momNextTick(callback: () => unknown): void;
interface NativeProcessInfo {
  execPath: string;
  arguments: string[];
  platform: NodeJS.Platform;
  arch: string;
  pid: number;
  ppid: number;
}
declare var __momProcessInfo: NativeProcessInfo;
declare function __momCwd(): string;
declare function __momChdir(directory: string): void;
declare function __momExitNow(code: number): never;
declare var __momNativeCore: NativeCoreModule | undefined;
declare function __momIsIP(value: string): number;
declare function __momHashCreate(): {update(data: Uint8Array): void, digest(): ArrayBuffer};
declare function __momRandomBytes(size: number): ArrayBuffer;
declare function __momGzip(data: Uint8Array): ArrayBuffer;
declare function __momCpus(): import("node:os").CpuInfo[];
declare function __momNvidiaComputeCapabilities(): number[];
declare function __momDisplayAdapters(): Array<{vendor: number, device: number, bus: number}>;
declare function __momReadFile(path: string): ArrayBuffer;
declare function __momFileExists(path: string): boolean;
declare function __momReadDirectory(path: string): string[];
declare function __momWrite(fd: number, bytes: Uint8Array): number;
declare function __momWriteFile(path: string, bytes: Uint8Array, mode: number, exclusive: boolean): void;
declare function __momChmod(path: string, mode: number): void;
declare function __momSpawn(args: string[], env: string[], callback: (event: string, value?: unknown) => void):
  {id: number, pid: number | undefined, stdin: number, stdout: number, stderr: number};
declare function __momProcessKill(id: number, signal: string): boolean;
declare function __momPipeRead(id: number, reading: boolean): void;
declare function __momStdinOpen(callback: (event: string, value?: unknown) => void): number;
declare function __momOutputOpen(fd: number, callback: (event: string, value?: unknown) => void): number | null;
declare function __momSignalStart(name: string, callback: () => unknown): number;
declare function __momImmediateStart(callback: () => unknown): number;
declare function __momNow(): number;
declare function __momRejectionCheckpoint(callback: () => boolean): void;
declare var HermesInternal: {
  enablePromiseRejectionTracker(options: {hostRejectionTracker(promise: object, reason: unknown, handled: boolean): void}): boolean;
};
declare function __momTLSConfigure(id: number, hostname: string, servername: string,
  rejectUnauthorized: boolean, ca: string | null): void;
declare function __momWorkerCreate(progress: (name: string, values: Record<string, string>) => void,
  complete: () => void, error: (error: Error) => void): {post(name: string, values: Array<[string, string]>): void};
