import type { DetailedHTMLProps, HTMLAttributes } from "react";

declare module "react" {
  namespace JSX {
    interface IntrinsicElements {
      "speech-engine-panel": DetailedHTMLProps<HTMLAttributes<HTMLElement>, HTMLElement>;
      "speech-models-panel": DetailedHTMLProps<HTMLAttributes<HTMLElement>, HTMLElement>;
      "speech-voice-storage": DetailedHTMLProps<HTMLAttributes<HTMLElement>, HTMLElement>;
      "llama-engine-panel": DetailedHTMLProps<HTMLAttributes<HTMLElement>, HTMLElement>;
      "llama-download-panel": DetailedHTMLProps<HTMLAttributes<HTMLElement>, HTMLElement>;
      "llama-models-panel": DetailedHTMLProps<HTMLAttributes<HTMLElement>, HTMLElement>;
      "logs-panel": DetailedHTMLProps<HTMLAttributes<HTMLElement>, HTMLElement>;
    }
  }
}