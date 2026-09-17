import type { DetailedHTMLProps, HTMLAttributes } from "react";

declare module "react" {
  namespace JSX {
    interface IntrinsicElements {
      "speech-engine-panel": DetailedHTMLProps<HTMLAttributes<HTMLElement>, HTMLElement>;
      "speech-models-panel": DetailedHTMLProps<HTMLAttributes<HTMLElement>, HTMLElement>;
      "speech-voice-storage": DetailedHTMLProps<HTMLAttributes<HTMLElement>, HTMLElement>;
    }
  }
}