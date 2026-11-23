// PDF handling for the execution agent's file tools. pi's read tool decodes
// any non-image file as UTF-8 text, which turns PDFs into binary garbage. Coco
// instead extracts PDF text with pypdf in its local Python service.
import fs from 'node:fs';
import {
  DEFAULT_MAX_BYTES,
  DEFAULT_MAX_LINES,
  truncateHead,
} from '@earendil-works/pi-coding-agent';

// PDF readers accept the header anywhere in the first 1 KB.
const PDF_HEADER_WINDOW = 1024;
// ripgrep globs are case-sensitive; a lone negated glob keeps all other files.
export const PDF_EXCLUDE_GLOB = '!*.[pP][dD][fF]';
// grep output rows: `path:line: text` for matches, `path-line- text` for context.
const PDF_GREP_ROW = /^(.+?\.pdf)(?::\d+: |-\d+- )/i;

const PDF_NOT_SEARCHED =
  "[PDF files were not searched. Use read to extract a PDF's text.]";

export const READ_PDF_NOTE =
  'PDFs are returned as extracted text, page by page; for PDFs, offset is the first page and limit is the number of pages.';

/**
 * Detect PDFs by content, not name: a mislabeled text file named `.pdf` must
 * stay readable, and extension-less PDFs must still be extracted.
 */
export async function isPdfFile(target) {
  let handle;
  try {
    handle = await fs.promises.open(target, 'r');
    const head = Buffer.alloc(PDF_HEADER_WINDOW);
    const { bytesRead } = await handle.read(head, 0, PDF_HEADER_WINDOW, 0);
    return head.subarray(0, bytesRead).includes('%PDF-', 0, 'latin1');
  } catch {
    return false;
  } finally {
    await handle?.close();
  }
}

/** Render extracted pages within pi's read output limits. */
export function formatPdfText(extraction) {
  const total = extraction.total_pages;
  if (!total) return { text: '[This PDF has no pages.]', shown: null };
  const header = `[PDF text extracted with pypdf: ${total} page${total === 1 ? '' : 's'}. Layout, tables and images may not be preserved.]`;
  const sections = extraction.pages.map(
    (page) => `--- Page ${page.number} of ${total} ---\n${page.text}`,
  );
  const fits = (text) =>
    !truncateHead(text, {
      maxLines: DEFAULT_MAX_LINES,
      maxBytes: DEFAULT_MAX_BYTES,
    }).truncated;

  let body = header;
  let shownCount = 0;
  for (const section of sections) {
    const next = `${body}\n\n${section}`;
    if (!fits(next)) break;
    body = next;
    shownCount += 1;
  }
  const first = extraction.first_page;
  if (shownCount === 0) {
    // A single page larger than the limit: show its beginning only.
    const partial = truncateHead(`${header}\n\n${sections[0]}`, {
      maxLines: DEFAULT_MAX_LINES,
      maxBytes: DEFAULT_MAX_BYTES,
    }).content;
    const more =
      first < total
        ? ` Use offset=${first + 1} to continue with page ${first + 1}.`
        : '';
    return {
      text: `${partial}\n\n[Page ${first} is longer than the read limit and was cut off.${more}]`,
      shown: { first, last: first, truncated: true },
    };
  }
  const last = first + shownCount - 1;
  if (last < total)
    body += `\n\n[Showing pages ${first}-${last} of ${total}. Use offset=${last + 1} to continue.]`;
  return {
    text: body,
    shown: { first, last, truncated: last < extraction.last_page },
  };
}

/**
 * Route PDFs to `extractPdf(bytes, { firstPage, lastPage }, signal)`; other
 * files keep pi's behavior. `checkRead` is the caller's permission check.
 */
export function pdfAwareRead(readTool, checkRead, extractPdf) {
  return {
    ...readTool,
    description: `${readTool.description} ${READ_PDF_NOTE}`,
    async execute(id, args, signal, update, ctx) {
      const target = checkRead(args.path);
      if (!(await isPdfFile(target)))
        return readTool.execute(id, args, signal, update, ctx);
      if (!extractPdf)
        throw new Error('Reading PDFs needs the local Coco service.');
      const data = await fs.promises.readFile(checkRead(target));
      const firstPage =
        args.offset === undefined ? 1 : Math.max(1, Math.floor(args.offset));
      const lastPage =
        args.limit === undefined
          ? undefined
          : firstPage + Math.max(1, Math.floor(args.limit)) - 1;
      const extraction = await extractPdf(
        data,
        { firstPage, lastPage },
        signal,
      );
      const { text, shown } = formatPdfText(extraction);
      return {
        content: [{ type: 'text', text }],
        details: { pdf: { totalPages: extraction.total_pages, shown } },
        structuredContent: text,
      };
    },
  };
}

/** Drop grep rows that come from PDF files, which ripgrep reads as binary. */
export function filterPdfGrepRows(result) {
  const block = result?.content?.[0];
  if (block?.type !== 'text') return result;
  const noticeStart = block.text.lastIndexOf('\n\n[');
  const body =
    noticeStart === -1 ? block.text : block.text.slice(0, noticeStart);
  const notices = noticeStart === -1 ? '' : block.text.slice(noticeStart);
  const skipped = new Set();
  const kept = body.split('\n').filter((row) => {
    const match = PDF_GREP_ROW.exec(row);
    if (match) skipped.add(match[1]);
    return !match;
  });
  if (skipped.size === 0) return result;
  const note = `[Skipped matches in ${skipped.size} PDF file${skipped.size === 1 ? '' : 's'}: grep cannot search PDF text. Use read to extract it.]`;
  const text = `${kept.length ? kept.join('\n') : 'No matches found'}${notices}\n\n${note}`;
  return { ...result, content: [{ ...block, text }] };
}

/** Keep PDF bytes out of grep output and point the model at read instead. */
export function pdfAwareGrep(grepTool, checkRead) {
  return {
    ...grepTool,
    description: `${grepTool.description} PDF files are skipped because their text is compressed; use read to extract PDF text.`,
    async execute(id, args, signal, update, ctx) {
      const target = checkRead(args.path ?? '.');
      const isFile = await fs.promises
        .stat(target)
        .then((stat) => stat.isFile())
        .catch(() => false); // missing paths keep pi's "Path not found" error
      if (isFile && (await isPdfFile(target)))
        throw new Error(
          `${args.path} is a PDF. grep cannot search PDF text; use read to extract it instead.`,
        );
      if (!args.glob) {
        const result = await grepTool.execute(
          id,
          { ...args, glob: PDF_EXCLUDE_GLOB },
          signal,
          update,
          ctx,
        );
        // Without this, "no matches" reads as "not in any file", PDFs included.
        return result?.content?.[0]?.text === 'No matches found'
          ? {
              ...result,
              content: [
                {
                  type: 'text',
                  text: `No matches found\n\n${PDF_NOT_SEARCHED}`,
                },
              ],
            }
          : result;
      }
      return filterPdfGrepRows(
        await grepTool.execute(id, args, signal, update, ctx),
      );
    },
  };
}

// Commands that print a PDF's raw bytes. Listing, copying, converting or
// writing PDFs (ls, find, cp, pdftotext, `> out.pdf`) is left alone.
const RAW_PDF_DUMP =
  /(?:^|[\s|;&(`])(?:cat|head|tail|less|more|strings|xxd|od|hexdump)\b[^|;&\n<>]*\.pdf\b/i;

export function dumpsRawPdf(command) {
  return typeof command === 'string' && RAW_PDF_DUMP.test(command);
}

/** Reject raw PDF dumps before approval so the user is never asked about them. */
export function guardRawPdfDumps(bashTool) {
  return {
    ...bashTool,
    async execute(id, args, signal, update, ctx) {
      if (dumpsRawPdf(args?.command))
        throw new Error(
          'This command prints raw PDF bytes, which are unreadable. Use the read tool on the PDF to get its extracted text.',
        );
      return bashTool.execute(id, args, signal, update, ctx);
    },
  };
}
