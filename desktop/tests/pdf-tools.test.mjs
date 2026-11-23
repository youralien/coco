import assert from 'node:assert/strict';
import { test } from 'node:test';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import http from 'node:http';
import { createPermissionTools } from '../release/app/agent/permission-tools.mjs';
import {
  dumpsRawPdf,
  filterPdfGrepRows,
  formatPdfText,
} from '../release/app/agent/pdf-tools.mjs';
import { createRuntime } from '../release/app/agent/pi-runtime.mjs';

// Compressed-looking PDF bytes without NUL, so ripgrep would treat them as text.
const FAKE_PDF = Buffer.from(
  '%PDF-1.7\n%\xe2\xe3\xcf\xd3\n1 0 obj\nstream\nx\x9c\xcbneedle\xf3\x8f\nendstream\n',
  'latin1',
);

function workspace(prefix) {
  const root = fs.realpathSync(fs.mkdtempSync(path.join(os.tmpdir(), prefix)));
  const cwd = path.join(root, 'workspace');
  fs.mkdirSync(cwd);
  return { root, cwd };
}

function toolsFor(cwd, extractPdf, requestApproval) {
  const tools = new Map(
    createPermissionTools(cwd, requestApproval, [], undefined, extractPdf).map(
      (tool) => [tool.name, tool],
    ),
  );
  return (name, args) =>
    Promise.resolve().then(() => tools.get(name).execute('call', args));
}

const pages = (first, last, total, text = (n) => `Text of page ${n}`) => ({
  total_pages: total,
  first_page: first,
  last_page: last,
  pages: Array.from({ length: last - first + 1 }, (_, i) => ({
    number: first + i,
    text: text(first + i),
  })),
});

test('read extracts PDFs through the service, keeps text files on pi, and maps offset/limit to pages', async () => {
  const { root, cwd } = workspace('coco-pdf-read-');
  fs.writeFileSync(path.join(cwd, 'report.pdf'), FAKE_PDF);
  fs.writeFileSync(path.join(cwd, 'scan'), FAKE_PDF); // PDF without extension
  fs.writeFileSync(path.join(cwd, 'notes.txt'), 'plain notes');
  fs.writeFileSync(path.join(cwd, 'mislabeled.pdf'), 'actually text');
  fs.writeFileSync(
    path.join(cwd, 'prefixed'),
    Buffer.concat([Buffer.from('junk before header\n'), FAKE_PDF]),
  );
  const calls = [];
  const extractPdf = async (data, range) => {
    calls.push({ data, range });
    return pages(range.firstPage, range.lastPage ?? 3, 3);
  };
  const execute = toolsFor(cwd, extractPdf);
  try {
    const full = await execute('read', { path: 'report.pdf' });
    assert.deepEqual(calls[0].data, FAKE_PDF);
    assert.deepEqual(calls[0].range, { firstPage: 1, lastPage: undefined });
    const text = full.content[0].text;
    assert.match(text, /extracted with pypdf: 3 pages/);
    assert.match(text, /--- Page 1 of 3 ---\nText of page 1/);
    assert.match(text, /--- Page 3 of 3 ---\nText of page 3/);
    assert.doesNotMatch(text, /%PDF|needle|Use offset/);
    assert.equal(full.structuredContent, text);

    const window = await execute('read', {
      path: 'report.pdf',
      offset: 2,
      limit: 1,
    });
    assert.deepEqual(calls[1].range, { firstPage: 2, lastPage: 2 });
    assert.match(window.content[0].text, /Page 2 of 3[\s\S]*Use offset=3/);

    await execute('read', { path: 'scan' });
    assert.equal(calls.length, 3, 'signature-detected PDFs are extracted');

    await execute('read', { path: 'prefixed' });
    assert.equal(calls.length, 4, 'a header within the first 1 KB counts');

    const notes = await execute('read', { path: 'notes.txt' });
    assert.match(JSON.stringify(notes), /plain notes/);
    const mislabeled = await execute('read', { path: 'mislabeled.pdf' });
    assert.match(JSON.stringify(mislabeled), /actually text/);
    assert.equal(calls.length, 4, 'text files never reach the PDF service');
    await assert.rejects(execute('read', { path: 'missing.pdf' }), /ENOENT/);
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test('PDF reads keep permission checks and fail clearly without the service', async () => {
  const { root, cwd } = workspace('coco-pdf-permission-');
  fs.writeFileSync(path.join(root, 'private.pdf'), FAKE_PDF);
  fs.writeFileSync(path.join(cwd, 'report.pdf'), FAKE_PDF);
  let extracted = 0;
  const execute = toolsFor(
    cwd,
    async () => {
      extracted++;
      return pages(1, 1, 1);
    },
    async () => false,
  );
  try {
    await assert.rejects(
      execute('read', { path: '../private.pdf' }),
      /Permission denied/,
    );
    assert.equal(extracted, 0, 'denied PDFs are never sent for extraction');
    await assert.rejects(
      toolsFor(cwd)('read', { path: 'report.pdf' }),
      /needs the local Coco service/,
    );
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test('PDF text is paged within read limits with continuation hints', () => {
  const large = (n) => `Page ${n} line\n`.repeat(600);
  const many = formatPdfText(pages(1, 10, 10, large));
  assert.equal(many.shown.first, 1);
  assert.ok(many.shown.last < 10);
  assert.match(
    many.text,
    new RegExp(`Use offset=${many.shown.last + 1} to continue`),
  );
  assert.ok(many.text.split('\n').length <= 2000 + 4);

  const huge = formatPdfText(pages(4, 4, 9, () => 'x'.repeat(80 * 1024)));
  assert.equal(huge.shown.truncated, true);
  assert.match(
    huge.text,
    /Page 4 is longer than the read limit[\s\S]*offset=5/,
  );
  assert.ok(Buffer.byteLength(huge.text) < 52 * 1024);

  assert.match(formatPdfText({ total_pages: 0, pages: [] }).text, /no pages/);
});

test('grep skips PDFs in directories, rejects direct PDF searches, and filters explicit globs', async () => {
  const { root, cwd } = workspace('coco-pdf-grep-');
  fs.writeFileSync(path.join(cwd, 'notes.txt'), 'a needle in notes\n');
  fs.writeFileSync(path.join(cwd, 'Paper.PDF'), FAKE_PDF);
  fs.mkdirSync(path.join(cwd, 'docs'));
  fs.writeFileSync(path.join(cwd, 'docs', 'deep.pdf'), FAKE_PDF);
  const execute = toolsFor(cwd);
  const output = async (args) => (await execute('grep', args)).content[0].text;
  try {
    const plain = await output({ pattern: 'needle' });
    assert.match(plain, /notes\.txt:1: a needle in notes/);
    assert.doesNotMatch(plain, /\.pdf|PDF/i);

    assert.equal(
      await output({ pattern: 'absent-term' }),
      "No matches found\n\n[PDF files were not searched. Use read to extract a PDF's text.]",
    );

    const context = await output({ pattern: 'needle', context: 1 });
    assert.doesNotMatch(context, /\.pdf/i);

    const globbed = await output({ pattern: 'needle', glob: '*' });
    assert.match(globbed, /notes\.txt:1: a needle in notes/);
    assert.doesNotMatch(globbed, /stream|�/);
    assert.match(globbed, /Skipped matches in 2 PDF files/);

    const onlyPdf = await output({ pattern: 'needle', glob: '*.pdf' });
    assert.match(
      onlyPdf,
      /^No matches found\n\n\[Skipped matches in 1 PDF file:/,
    );

    await assert.rejects(
      execute('grep', { pattern: 'needle', path: 'docs/deep.pdf' }),
      /is a PDF\. grep cannot search PDF text; use read/,
    );
    await assert.rejects(
      execute('grep', { pattern: 'needle', path: 'missing' }),
      /Path not found/,
    );
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test('grep row filtering keeps notices and ignores non-PDF paths', () => {
  const result = (text) => ({ content: [{ type: 'text', text }] });
  const filtered = filterPdfGrepRows(
    result(
      'a.pdf:3: âgarbage\nnotes.pdf-archive/x.txt:2: real\nb.pdf-4- ctx\n\n[100 matches limit reached]',
    ),
  ).content[0].text;
  assert.equal(
    filtered,
    'notes.pdf-archive/x.txt:2: real\n\n[100 matches limit reached]\n\n[Skipped matches in 2 PDF files: grep cannot search PDF text. Use read to extract it.]',
  );
  const untouched = result('src/a.ts:1: ok');
  assert.equal(filterPdfGrepRows(untouched), untouched);
});

test('bash rejects raw PDF dumps before asking for approval', async () => {
  for (const command of [
    'cat report.pdf',
    'head -c 2000 "My Paper.PDF"',
    'ls && strings docs/a.pdf | head',
    'xxd scan.pdf',
  ])
    assert.equal(dumpsRawPdf(command), true, command);
  for (const command of [
    'ls *.pdf | head',
    'find . -name "*.pdf"',
    'cp a.pdf b.pdf',
    'python3 extract.py report.pdf',
    'pdftotext report.pdf - | head -50',
    'cat notes.txt',
    'cat cover.txt > out.pdf',
  ])
    assert.equal(dumpsRawPdf(command), false, command);

  const { root, cwd } = workspace('coco-pdf-bash-');
  let approvals = 0;
  const execute = toolsFor(cwd, undefined, async () => {
    approvals++;
    return true;
  });
  try {
    await assert.rejects(
      execute('bash', { command: 'cat report.pdf' }),
      /Use the read tool on the PDF/,
    );
    assert.equal(approvals, 0);
    const listed = await execute('bash', { command: 'echo ok' });
    assert.match(JSON.stringify(listed), /ok/);
    assert.equal(approvals, 1);
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test('the runtime sends PDF bytes to the extract endpoint and gives the model text', async () => {
  const { root, cwd } = workspace('coco-pdf-runtime-');
  fs.mkdirSync(path.join(root, 'pi'));
  fs.writeFileSync(path.join(root, 'pi/browser.json'), '{"enabled":false}');
  fs.writeFileSync(path.join(cwd, 'report.pdf'), FAKE_PDF);
  const chats = [];
  let extractRequest;
  const server = http.createServer(async (req, res) => {
    assert.equal(req.headers.authorization, 'Bearer test-token');
    const chunks = [];
    for await (const chunk of req) chunks.push(chunk);
    const body = Buffer.concat(chunks);
    if (req.url.startsWith('/agent/pdf/extract')) {
      extractRequest = {
        url: req.url,
        type: req.headers['content-type'],
        body,
      };
      res.setHeader('Content-Type', 'application/json');
      res.end(JSON.stringify(pages(1, 2, 2, (n) => `Revenue grew ${n}0%`)));
      return;
    }
    chats.push(JSON.parse(body.toString()));
    const delta =
      chats.length === 1
        ? {
            role: 'assistant',
            tool_calls: [
              {
                index: 0,
                id: 'read-1',
                type: 'function',
                function: {
                  name: 'read',
                  arguments: '{"path":"report.pdf"}',
                },
              },
            ],
          }
        : { role: 'assistant', content: 'Revenue grew.' };
    res.writeHead(200, { 'Content-Type': 'text/event-stream' });
    res.end(
      `data: ${JSON.stringify({ id: 't', object: 'chat.completion.chunk', created: 1, model: 'coco', choices: [{ index: 0, delta, finish_reason: chats.length === 1 ? 'tool_calls' : 'stop' }] })}\n\ndata: [DONE]\n\n`,
    );
  });
  await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
  const context = {
    instructions: 'You are Coco.',
    model: 'test',
    profile: '',
    history: [],
    tools: [],
  };
  let runtime;
  try {
    runtime = await createRuntime({
      cwd,
      dataDir: root,
      sessionId: 'pdf',
      tutorUrl: `http://127.0.0.1:${server.address().port}`,
      token: 'test-token',
      context,
      requestWorkspaceAccess: async () => true,
    });
    await runtime.prompt('Summarize report.pdf', [], context);
    const first = JSON.stringify(chats[0]);
    assert.match(first, /Read PDFs with the read tool/);
    const read = chats[0].tools.find((t) => t.function.name === 'read');
    assert.match(
      read.function.description,
      /PDFs are returned as extracted text/,
    );

    assert.equal(extractRequest.url, '/agent/pdf/extract?first_page=1');
    assert.equal(extractRequest.type, 'application/pdf');
    assert.deepEqual(extractRequest.body, FAKE_PDF);

    const toolResult = JSON.stringify(chats[1].messages.at(-1));
    assert.match(toolResult, /Page 1 of 2.*Revenue grew 10%/);
    assert.doesNotMatch(toolResult, /%PDF|needle/);
  } finally {
    await runtime?.dispose();
    await new Promise((resolve) => server.close(resolve));
    fs.rmSync(root, { recursive: true, force: true });
  }
});
