import { useEffect, useRef, useState } from 'react';

function CopyIcon() {
  return (
    <svg
      viewBox="0 0 24 24"
      width="15"
      height="15"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden
    >
      <rect x="9" y="9" width="12" height="12" rx="2" />
      <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
    </svg>
  );
}

function CheckIcon() {
  return (
    <svg
      viewBox="0 0 24 24"
      width="15"
      height="15"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden
    >
      <path d="M20 6 9 17l-5-5" />
    </svg>
  );
}

export interface CopyPromptFeedback {
  surface: 'bubble' | 'notification' | 'chat';
  observationId?: string | null;
  status?: string | null;
  sessionId?: string | null;
  latencyS?: number | null;
}

/**
 * Copy a suggested prompt to the clipboard. Copying is logged as an implicit
 * positive ("copy") feedback event, so personalization treats it as a sign the
 * suggestion was useful. The clipboard write goes through main so it works in
 * unfocused overlay windows (pet bubble, pop-up), where navigator.clipboard
 * is rejected.
 */
export function copySuggestedPrompt(
  text: string,
  feedback: CopyPromptFeedback,
) {
  const ipc = window.electron?.ipcRenderer;
  ipc?.sendMessage('suggestion-action', { copyText: text });
  ipc?.sendMessage('training-feedback', {
    kind: 'copy',
    surface: feedback.surface,
    observation_id: feedback.observationId ?? null,
    status: feedback.status ?? null,
    ...(feedback.sessionId ? { session_id: feedback.sessionId } : {}),
    latency_s: feedback.latencyS ?? null,
    text,
  });
}

/** Icon button shown beside "Run with Coco" for delegation prompts. */
export default function CopyPromptButton({
  text,
  getFeedback,
}: {
  text: string;
  /** Read at click time so latency reflects when the user copied. */
  getFeedback: () => CopyPromptFeedback;
}) {
  const [copied, setCopied] = useState(false);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  useEffect(
    () => () => {
      if (timer.current) clearTimeout(timer.current);
    },
    [],
  );
  if (!text) return null;
  const label = copied ? 'Prompt copied' : 'Copy prompt';
  return (
    <button
      type="button"
      className={`coco-rating-btn coco-copy-btn${copied ? ' is-copied' : ''}`}
      aria-label={label}
      title={label}
      onClick={() => {
        copySuggestedPrompt(text, getFeedback());
        setCopied(true);
        if (timer.current) clearTimeout(timer.current);
        timer.current = setTimeout(() => setCopied(false), 2000);
      }}
    >
      {copied ? <CheckIcon /> : <CopyIcon />}
    </button>
  );
}
