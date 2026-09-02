/**
 * Top-down decision tree renderer for the preference model.
 * SURA 2026 · IIT Delhi
 *
 * The model is a Fast-and-Frugal Tree: an ordered list of checks where each
 * check either exits to an outcome (YES) or falls through to the next check
 * (NO). That structure IS a binary decision tree, but the previous drawing
 * showed it as a left-hand column of checks with exits trailing off to the
 * right, which read as a list with loose ends rather than a tree.
 *
 * This renderer draws the same model the conventional way: the first check at
 * the top, YES branching down-left to its outcome, NO continuing down-right to
 * the next check, and the fall-through outcome at the bottom of the spine. A
 * tie-breaker on a check becomes a sub-check under that check's YES branch,
 * with its own two outcomes — so it reads as part of the tree instead of a
 * detached box.
 *
 * Layout is a standard two-pass tidy-tree: measure every subtree's width
 * bottom-up, then place each parent centred over its two children. Nothing
 * about the model's meaning changes — only where the boxes sit.
 *
 * Usage:
 *     var out = FFTTree.build(tree, {
 *       editing: false,
 *       selectedNode: null,
 *       explanations: [{explanation, refine_explanation}, ...],
 *       helpers: {...}     // see REQUIRED HELPERS below
 *     });
 *     container.innerHTML = out.svg;
 *     out.layout            // box geometry, for positioning HTML panels
 *
 * REQUIRED HELPERS (supplied by the caller so label wording stays in one place):
 *     esc(s)                        HTML-escape
 *     fmt(x)                        number -> display string
 *     wrapText(text, maxChars)      greedy word wrap -> array of lines
 *     prettyFeature(f)              "age_diff" -> "Age(A) - Age(B)"
 *     outcomeLabel(node, short)     outcome text for a check's YES exit
 *     refineOutcomeLabel(r, isTrue) outcome text for a tie-breaker branch
 *     defaultOutcomeLabel()         outcome text for the fall-through leaf
 */
(function (global) {
  'use strict';

  // ── geometry ───────────────────────────────────────────────────────────────
  var PAD        = 26;   // canvas margin
  var NODE_W     = 268;  // decision box width
  var NODE_MIN_H = 66;
  var LEAF_W     = 158;
  var LEAF_H     = 52;
  var H_GAP      = 26;   // between sibling subtrees
  var V_GAP      = 52;   // between levels (room for branch labels)

  var CAP_CHARS  = 30;   // explanation wrap width inside a decision box
  var CAP_LINE_H = 16;
  var COND_CHARS = 32;
  var COND_LINE_H = 14;
  var LEAF_CHARS = 15;
  var LEAF_LINE_H = 15;

  var C = {
    text: '#0f172a', card: '#f8fafc', border: '#e2e8f0',
    dim: '#64748b', muted: '#94a3b8', accent: '#2563eb',
    a: '#b91c1c', b: '#1d4ed8'
  };

  function leafFill(cls) { return cls === 1 ? C.a : C.b; }

  // ── model -> binary tree ───────────────────────────────────────────────────
  //
  // Check i:  YES -> outcome (or tie-breaker sub-check),  NO -> check i+1.
  // The last check's NO branch is the fall-through ("OTHERWISE") outcome.

  function toBinary(tree, H) {
    var nodes = (tree && tree.nodes) || [];

    function defaultLeaf() {
      return {
        kind: 'leaf', role: 'default',
        cls: tree.default_class === 1 ? 1 : 0,
        lines: H.wrapText(H.defaultOutcomeLabel(), LEAF_CHARS),
        eyebrow: 'OTHERWISE'
      };
    }

    function yesBranch(nd, i, exp) {
      if (!nd.refine) {
        // For abs-value checks the colour comes from prefer_higher; older
        // signed checks still carry it in exit_class.
        var cls = nd.use_abs ? (nd.prefer_higher !== false ? 1 : 0) : nd.exit_class;
        return {
          kind: 'leaf', role: 'exit', nodeIndex: i, cls: cls,
          lines: H.wrapText(H.outcomeLabel(nd, true), LEAF_CHARS)
        };
      }
      var r = nd.refine;
      var rName = (r.feature || '').replace('_diff', '').replace(/_/g, ' ')
        .replace(/\b\w/g, function (c) { return c.toUpperCase(); });
      return {
        kind: 'decision', role: 'refine', nodeIndex: i,
        capLines: H.wrapText((exp && exp.refine_explanation) || 'Closer look', CAP_CHARS),
        condLines: H.wrapText(
          '|\u0394 ' + rName + '| ' + (r.op === '<=' ? '\u2264' : '\u2265') +
          ' ' + H.fmt(Math.abs(r.threshold)), COND_CHARS),
        eyebrow: 'TIE-BREAKER',
        yes: {
          kind: 'leaf', role: 'refine-exit', nodeIndex: i, branch: true,
          cls: r.true_class, lines: H.wrapText(H.refineOutcomeLabel(r, true), LEAF_CHARS)
        },
        no: {
          kind: 'leaf', role: 'refine-exit', nodeIndex: i, branch: false,
          cls: r.false_class, lines: H.wrapText(H.refineOutcomeLabel(r, false), LEAF_CHARS)
        }
      };
    }

    function checkAt(i) {
      if (i >= nodes.length) return defaultLeaf();
      var nd = nodes[i];
      var exp = (H.explanations && H.explanations[i]) || {};
      var condText = nd.refine
        ? ('|' + H.prettyFeature(nd.feature) + '| ' +
           (nd.op === '<=' ? '\u2264' : '\u2265') + ' ' + H.fmt(Math.abs(nd.threshold)))
        : H.nodeCondText(nd);
      return {
        kind: 'decision', role: 'check', nodeIndex: i,
        eyebrow: 'STEP ' + (i + 1),
        capLines: H.wrapText(exp.explanation, CAP_CHARS),
        condLines: H.wrapText(condText, COND_CHARS),
        yes: yesBranch(nd, i, exp),
        no: checkAt(i + 1)
      };
    }

    return nodes.length ? checkAt(0) : defaultLeaf();
  }

  // ── measure ────────────────────────────────────────────────────────────────

  function measure(n) {
    if (n.kind === 'leaf') {
      n.w = LEAF_W;
      n.h = Math.max(LEAF_H, 20 + n.lines.length * LEAF_LINE_H +
                             (n.eyebrow ? 14 : 0));
      n.subW = n.w;
      return;
    }
    n.w = NODE_W;
    n.h = Math.max(
      NODE_MIN_H,
      30 + n.capLines.length * CAP_LINE_H + n.condLines.length * COND_LINE_H + 14
    );
    measure(n.yes);
    measure(n.no);
    n.subW = Math.max(n.w, n.yes.subW + H_GAP + n.no.subW);
  }

  // ── place ──────────────────────────────────────────────────────────────────

  function place(n, left, depth, rowH) {
    n.depth = depth;
    rowH[depth] = Math.max(rowH[depth] || 0, n.h);
    if (n.kind === 'leaf') {
      n.cx = left + n.subW / 2;
      return;
    }
    place(n.yes, left, depth + 1, rowH);
    place(n.no, left + n.yes.subW + H_GAP, depth + 1, rowH);
    n.cx = (n.yes.cx + n.no.cx) / 2;
  }

  function assignY(n, rowY) {
    n.y = rowY[n.depth];
    n.x = n.cx - n.w / 2;
    if (n.kind === 'decision') { assignY(n.yes, rowY); assignY(n.no, rowY); }
  }

  // ── draw ───────────────────────────────────────────────────────────────────

  function badge(cx, cy, r, glyph, fill, fontSize, attrs, title) {
    return '<g' + (attrs ? ' ' + attrs : '') + ' style="cursor:pointer">' +
      (title ? '<title>' + title + '</title>' : '') +
      '<circle cx="' + cx + '" cy="' + cy + '" r="' + r + '" fill="' + fill + '"/>' +
      '<text x="' + cx + '" y="' + (cy + fontSize * 0.35) + '" font-size="' + fontSize +
      '" text-anchor="middle" fill="#fff" font-weight="700" ' +
      'style="pointer-events:none">' + glyph + '</text></g>';
  }

  function elbow(x1, y1, x2, y2) {
    // Vertical drop, horizontal run, vertical drop into the child's top edge.
    var mid = y1 + (y2 - y1) / 2;
    return 'M' + x1 + ',' + y1 + ' L' + x1 + ',' + mid +
           ' L' + x2 + ',' + mid + ' L' + x2 + ',' + (y2 - 5);
  }

  function drawEdges(s, n, H) {
    if (n.kind !== 'decision') return;
    [['yes', n.yes, 'YES'], ['no', n.no, 'NO']].forEach(function (pair) {
      var child = pair[1];
      var x1 = n.cx, y1 = n.y + n.h, x2 = child.cx, y2 = child.y;
      s.push('<path d="' + elbow(x1, y1, x2, y2) + '" fill="none" stroke="' +
        C.muted + '" stroke-width="1.4" marker-end="url(#ah)"/>');
      var lx = x2 + (x2 < x1 ? -10 : 10);
      var anchor = x2 < x1 ? 'end' : 'start';
      if (Math.abs(x2 - x1) < 4) { lx = x1 + 10; anchor = 'start'; }
      s.push('<text x="' + lx + '" y="' + (y1 + (y2 - y1) / 2 - 6) +
        '" font-size="11.5" text-anchor="' + anchor + '" fill="' + C.muted +
        '" font-weight="700" letter-spacing=".05em">' + pair[2] + '</text>');
    });
    drawEdges(s, n.yes, H);
    drawEdges(s, n.no, H);
  }

  function drawNode(s, n, opts, H, layout) {
    var editing = opts.editing;

    if (n.kind === 'leaf') {
      var f = leafFill(n.cls);
      var cur = editing ? ' style="cursor:pointer"' : '';
      var attr = '';
      if (n.role === 'exit') attr = 'data-leaf="' + n.nodeIndex + '"';
      else if (n.role === 'default') attr = 'id="default-leaf"';
      else if (n.role === 'refine-exit') attr = 'data-refine="' + n.nodeIndex + '"';

      s.push('<g ' + attr + cur + '>');
      ['fill="' + f + '" opacity=".15"',
       'fill="none" stroke="' + f + '" stroke-width="' + (editing ? '1.8' : '1.4') + '"' +
         (editing ? ' stroke-dasharray="4 3"' : '')
      ].forEach(function (style) {
        s.push('<rect x="' + n.x + '" y="' + n.y + '" rx="10" width="' + n.w +
          '" height="' + n.h + '" ' + style + '/>');
      });

      var top = n.y + (n.eyebrow ? 26 : 18);
      if (n.eyebrow) {
        s.push('<text x="' + (n.x + n.w / 2) + '" y="' + (n.y + 16) +
          '" font-size="9.5" text-anchor="middle" fill="' + f +
          '" opacity=".75" font-weight="700" letter-spacing=".06em">' +
          n.eyebrow + '</text>');
      }
      var startY = top + (n.h - (n.eyebrow ? 26 : 18) - n.lines.length * LEAF_LINE_H) / 2;
      n.lines.forEach(function (line, li) {
        s.push('<text x="' + (n.x + n.w / 2) + '" y="' + (startY + li * LEAF_LINE_H) +
          '" font-size="13" text-anchor="middle" fill="' + f +
          '" font-weight="700">' + H.esc(line) + '</text>');
      });
      if (editing) {
        s.push(badge(n.x + n.w - 13, n.y + 13, 10, '\u270E', C.accent, 11, '',
          'Change this outcome'));
      }
      s.push('</g>');

      if (n.role === 'default') layout.defaultLeaf = {x: n.x, y: n.y, w: n.w, h: n.h};
      else if (n.role === 'exit') layout.leaves[n.nodeIndex] = {x: n.x, y: n.y, w: n.w, h: n.h};
      return;
    }

    // Decision box
    var isRefine = n.role === 'refine';
    var sel = editing && !isRefine && opts.selectedNode === n.nodeIndex;
    var stroke = sel ? C.accent : (editing ? C.accent : C.border);
    var sw = sel ? '2.5' : (editing ? '1.6' : '1.2');
    var groupAttr = isRefine
      ? 'data-refine="' + n.nodeIndex + '"'
      : 'data-node="' + n.nodeIndex + '"';
    var cursor = editing ? ' style="cursor:pointer"' : '';

    s.push('<g ' + groupAttr + cursor + '>');
    s.push('<rect x="' + n.x + '" y="' + n.y + '" rx="11" width="' + n.w + '" height="' +
      n.h + '" fill="' + C.card + '" stroke="' + stroke + '" stroke-width="' + sw + '"' +
      (editing || isRefine ? ' stroke-dasharray="5 3"' : '') + '/>');
    s.push('<text x="' + (n.x + (editing && !isRefine ? 38 : 14)) + '" y="' + (n.y + 19) +
      '" font-size="11.5" fill="' + C.muted + '" font-weight="700" ' +
      'letter-spacing=".06em">' + n.eyebrow + '</text>');

    var ty = n.y + 36;
    n.capLines.forEach(function (line, li) {
      s.push('<text x="' + (n.x + 14) + '" y="' + (ty + li * CAP_LINE_H) +
        '" font-size="12.5" fill="' + C.text + '" font-weight="600">' +
        H.esc(line) + '</text>');
    });
    var my = ty + n.capLines.length * CAP_LINE_H + 4;
    n.condLines.forEach(function (line, li) {
      s.push('<text x="' + (n.x + 14) + '" y="' + (my + li * COND_LINE_H) +
        '" font-size="11.5" fill="' + C.dim + '" font-family="monospace">' +
        H.esc(line) + '</text>');
    });
    s.push('</g>');

    if (isRefine) {
      layout.refines[n.nodeIndex] = {x: n.x, y: n.y, w: n.w, h: n.h};
      if (editing) {
        s.push(badge(n.x + n.w - 13, n.y + 13, 10, '\u270E', C.accent, 11,
          'data-refine="' + n.nodeIndex + '"', 'Edit this tie-breaker'));
      }
    } else {
      layout.nodes[n.nodeIndex] = {x: n.x, y: n.y, w: n.w, h: n.h, cx: n.cx};
      if (editing) {
        var i = n.nodeIndex;
        s.push(badge(n.x + 19, n.y + 16, 12, '\u270E', C.accent, 13,
          'data-node="' + i + '"', 'Edit this check'));
        // One-click insert. Sits on the box itself so adding a check never
        // requires opening a panel first.
        var addFill = opts.canAddMore === false ? C.muted : '#15803d';
        var addTitle = opts.canAddMore === false
          ? 'Limit reached — you can add two checks at a time'
          : 'Add a check straight after this one';
        s.push(badge(n.x + n.w - 19, n.y + n.h - 1, 13, '+', addFill, 17,
          'data-add-after="' + i + '"', addTitle));
        if (i > 0) {
          s.push(badge(n.x + n.w - 46, n.y + 16, 11, '\u25B2', C.text, 12,
            'data-move-up="' + i + '"', 'Move this check earlier'));
        }
        if (opts.nodeCount && i < opts.nodeCount - 1) {
          s.push(badge(n.x + n.w - 20, n.y + 16, 11, '\u25BC', C.text, 12,
            'data-move-down="' + i + '"', 'Move this check later'));
        }
      }
    }

    drawNode(s, n.yes, opts, H, layout);
    drawNode(s, n.no, opts, H, layout);
  }

  // ── public API ─────────────────────────────────────────────────────────────

  function build(tree, opts) {
    opts = opts || {};
    var H = opts.helpers || {};
    H.explanations = opts.explanations || [];

    var root = toBinary(tree, H);
    measure(root);

    var rowH = [];
    place(root, PAD, 0, rowH);

    var rowY = [], acc = PAD;
    for (var d = 0; d < rowH.length; d++) {
      rowY[d] = acc;
      acc += rowH[d] + V_GAP;
    }
    assignY(root, rowY);

    var W = root.subW + PAD * 2;
    var H_TOTAL = acc - V_GAP + PAD;

    var layout = {nodes: {}, leaves: {}, refines: {}, defaultLeaf: null,
                  width: W, height: H_TOTAL};

    var s = [];
    s.push('<svg viewBox="0 0 ' + W + ' ' + H_TOTAL + '" ' +
      'xmlns="http://www.w3.org/2000/svg" ' +
      'font-family="-apple-system,Segoe UI,Roboto,sans-serif">');
    s.push('<defs><marker id="ah" markerWidth="9" markerHeight="9" refX="6" refY="3" ' +
      'orient="auto"><path d="M0,0 L7,3 L0,6 Z" fill="' + C.muted +
      '"/></marker></defs>');

    drawEdges(s, root, H);
    drawNode(s, root, opts, H, layout);
    s.push('</svg>');

    return {svg: s.join(''), layout: layout, width: W, height: H_TOTAL};
  }

  global.FFTTree = {build: build, COLORS: C};
})(window);
