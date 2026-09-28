/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** Base URL of a MouthTwin inference server (see server/). When unset, uploads use the in-browser template mapper. */
  readonly VITE_INFERENCE_URL?: string
}
