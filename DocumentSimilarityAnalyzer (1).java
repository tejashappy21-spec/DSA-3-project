/*
 * ============================================================================
 *  Document Similarity Analyzer Using Suffix Structures and Similarity Metrics
 *  Data Structures and Algorithms - 3 (25CS2103E)   |   A.Y. 2026-2027
 *  Team: Harsha Charan (2520030200), Ram Charan (2520030318), Tejas (2520030407)
 * ============================================================================
 *
 *  PIPELINE (same as the flow chart in the PPT)
 *    Input documents -> Preprocess -> Build Suffix Array (+ LCP array)
 *      -> Find common substrings -> Similarity score -> Compare all pairs
 *      -> Rank documents -> Display results
 *
 *  HOW IT WORKS
 *   1. Preprocessing : lower-case, keep only letters/digits, collapse spaces.
 *   2. For every pair (A, B) we build ONE suffix array over  A + '#' + B.
 *      '#' never occurs in the cleaned text, so no match can cross the border.
 *   3. Kasai's algorithm builds the LCP array in O(n).
 *   4. Walking the suffix array once (forwards and backwards) gives, for every
 *      position of A, the longest substring starting there that also occurs in
 *      B (and vice versa).   -> no character-by-character comparison of docs.
 *   5. Every match of length >= "minimum match length" marks its characters as
 *      "covered".  Similarity = covered chars of both docs / total chars.
 *   6. Pairs are sorted by score (highest first) and classified.
 *
 *  COMPLEXITY (n = |A| + |B|)
 *      Suffix array : O(n log n)   Kasai LCP : O(n)   Matching : O(n)
 *      Naive substring comparison would be O(|A| * |B| * min(|A|,|B|)).
 *
 *  REQUIREMENTS : Java 8 or newer, standard libraries only (no downloads).
 *
 *  RUN
 *      javac DocumentSimilarityAnalyzer.java
 *      java  DocumentSimilarityAnalyzer                    -> opens the GUI
 *      java  DocumentSimilarityAnalyzer a.txt b.txt c.txt  -> command line
 *      java  DocumentSimilarityAnalyzer --demo             -> built-in demo
 *      Options: --min=20 (min match length)  --details (show matching passages)
 *  (Java 11+ can also run it directly:  java DocumentSimilarityAnalyzer.java)
 * ============================================================================
 */

import javax.swing.*;
import javax.swing.event.DocumentEvent;
import javax.swing.event.DocumentListener;
import javax.swing.filechooser.FileNameExtensionFilter;
import javax.swing.table.DefaultTableCellRenderer;
import javax.swing.table.DefaultTableModel;
import java.awt.*;
import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.charset.CharacterCodingException;
import java.nio.charset.CodingErrorAction;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.function.IntConsumer;

public class DocumentSimilarityAnalyzer {

    // =========================================================================
    //  1. DATA MODEL
    // =========================================================================

    /** One input document: original text + cleaned text used for comparison. */
    static final class Doc {
        final String name;
        final String raw;
        final String text;   // preprocessed

        Doc(String name, String raw) {
            this.name = name;
            this.raw = raw;
            this.text = Preprocessor.normalize(raw);
        }

        @Override
        public String toString() {
            return name + "  (" + text.length() + " chars)";
        }
    }

    /** Result of comparing one pair of documents. */
    static final class PairResult {
        Doc a, b;
        int ai, bi;                 // indices in the analysed list
        double similarity;          // 0..100
        double covA, covB;          // % of A / B covered by common substrings
        int lcsLen;                 // longest common substring length
        String lcsText = "";
        String label = "";
        List<String> passages = new ArrayList<String>();
        long nanos;                 // time taken for this pair
    }

    // =========================================================================
    //  2. PREPROCESSING  (normalisation, case conversion, junk removal)
    // =========================================================================

    static final class Preprocessor {
        static String normalize(String raw) {
            StringBuilder sb = new StringBuilder(raw.length());
            boolean lastSpace = true;
            for (int i = 0; i < raw.length(); i++) {
                char c = Character.toLowerCase(raw.charAt(i));
                if (Character.isLetterOrDigit(c)) {
                    sb.append(c);
                    lastSpace = false;
                } else if (!lastSpace) {
                    sb.append(' ');
                    lastSpace = true;
                }
            }
            int n = sb.length();
            if (n > 0 && sb.charAt(n - 1) == ' ') sb.setLength(n - 1);
            return sb.toString();
        }
    }

    // =========================================================================
    //  3. SUFFIX ARRAY  (prefix doubling + counting sort, O(n log n))
    // =========================================================================

    /** Returns SA where SA[i] = start index of the i-th smallest suffix. */
    static int[] buildSuffixArray(String str) {
        int len = str.length();
        int n = len + 1;                       // +1 for a unique smallest sentinel
        boolean[] present = new boolean[65536];
        for (int i = 0; i < len; i++) present[str.charAt(i)] = true;
        int[] code = new int[65536];
        int alphabet = 1;                      // code 0 is reserved for the sentinel
        for (int ch = 0; ch < 65536; ch++) if (present[ch]) code[ch] = alphabet++;

        int[] s = new int[n];
        for (int i = 0; i < len; i++) s[i] = code[str.charAt(i)];
        s[len] = 0;

        int[] p = new int[n], c = new int[n], pn = new int[n], cn = new int[n];
        int[] cnt = new int[Math.max(alphabet, n) + 1];

        for (int i = 0; i < n; i++) cnt[s[i]]++;
        for (int i = 1; i < alphabet; i++) cnt[i] += cnt[i - 1];
        for (int i = n - 1; i >= 0; i--) p[--cnt[s[i]]] = i;

        c[p[0]] = 0;
        int classes = 1;
        for (int i = 1; i < n; i++) {
            if (s[p[i]] != s[p[i - 1]]) classes++;
            c[p[i]] = classes - 1;
        }

        for (int h = 0; (1 << h) < n && classes < n; h++) {
            int shift = 1 << h;
            for (int i = 0; i < n; i++) {
                pn[i] = p[i] - shift;
                if (pn[i] < 0) pn[i] += n;
            }
            Arrays.fill(cnt, 0, classes, 0);
            for (int i = 0; i < n; i++) cnt[c[pn[i]]]++;
            for (int i = 1; i < classes; i++) cnt[i] += cnt[i - 1];
            for (int i = n - 1; i >= 0; i--) p[--cnt[c[pn[i]]]] = pn[i];

            cn[p[0]] = 0;
            classes = 1;
            for (int i = 1; i < n; i++) {
                int a1 = c[p[i]], a2 = c[(p[i] + shift) % n];
                int b1 = c[p[i - 1]], b2 = c[(p[i - 1] + shift) % n];
                if (a1 != b1 || a2 != b2) classes++;
                cn[p[i]] = classes - 1;
            }
            int[] tmp = c; c = cn; cn = tmp;
        }
        return Arrays.copyOfRange(p, 1, n);    // drop the sentinel suffix
    }

    /** Kasai's algorithm. lcp[i] = LCP(suffix SA[i-1], suffix SA[i]); lcp[0] = 0. */
    static int[] buildLcp(String s, int[] sa) {
        int n = s.length();
        int[] rank = new int[n];
        for (int i = 0; i < n; i++) rank[sa[i]] = i;
        int[] lcp = new int[n];
        int h = 0;
        for (int i = 0; i < n; i++) {
            if (rank[i] > 0) {
                int j = sa[rank[i] - 1];
                while (i + h < n && j + h < n && s.charAt(i + h) == s.charAt(j + h)) h++;
                lcp[rank[i]] = h;
                if (h > 0) h--;
            } else {
                h = 0;
            }
        }
        return lcp;
    }

    // =========================================================================
    //  4. ANALYZER  (common substrings -> similarity score -> ranking)
    // =========================================================================

    static String classify(double similarity) {
        if (similarity >= 95) return "Near-Duplicate";
        if (similarity >= 80) return "Highly Similar";
        if (similarity >= 40) return "Moderately Similar";
        return "Low Similarity";
    }

    /** Compare two documents using a generalised suffix array over A#B. */
    static PairResult compare(Doc A, Doc B, int minLen) {
        long t0 = System.nanoTime();
        PairResult r = new PairResult();
        r.a = A;
        r.b = B;
        final String ta = A.text, tb = B.text;
        final int la = ta.length(), lb = tb.length();
        if (la == 0 || lb == 0) {
            r.label = classify(0);
            r.nanos = System.nanoTime() - t0;
            return r;
        }

        String t = ta + '#' + tb;                 // '#' is removed by preprocessing
        int[] sa = buildSuffixArray(t);
        int[] lcp = buildLcp(t, sa);
        int m = sa.length;

        // match[p] = longest substring starting at p that occurs in the OTHER document
        int[] match = new int[t.length()];
        final int INF = Integer.MAX_VALUE;

        // forward pass: nearest suffix ABOVE in the suffix array from the other doc
        int runA = 0, runB = 0;
        for (int i = 0; i < m; i++) {
            if (i > 0) { runA = Math.min(runA, lcp[i]); runB = Math.min(runB, lcp[i]); }
            int p = sa[i];
            if (p < la)      { match[p] = Math.max(match[p], runB); runA = INF; }
            else if (p > la) { match[p] = Math.max(match[p], runA); runB = INF; }
        }
        // backward pass: nearest suffix BELOW
        runA = 0; runB = 0;
        for (int i = m - 1; i >= 0; i--) {
            if (i < m - 1) { runA = Math.min(runA, lcp[i + 1]); runB = Math.min(runB, lcp[i + 1]); }
            int p = sa[i];
            if (p < la)      { match[p] = Math.max(match[p], runB); runA = INF; }
            else if (p > la) { match[p] = Math.max(match[p], runA); runB = INF; }
        }

        // coverage via difference arrays
        int[] dA = new int[la + 1], dB = new int[lb + 1];
        int lcsLen = 0, lcsPos = -1;
        for (int p = 0; p < la; p++) {
            int L = match[p];
            if (L > lcsLen) { lcsLen = L; lcsPos = p; }
            if (L >= minLen) { dA[p]++; dA[p + L]--; }
        }
        for (int q = 0; q < lb; q++) {
            int L = match[la + 1 + q];
            if (L >= minLen) { dB[q]++; dB[q + L]--; }
        }
        boolean[] covA = new boolean[la];
        boolean[] covB = new boolean[lb];
        int cntA = 0, cntB = 0, run = 0;
        for (int p = 0; p < la; p++) { run += dA[p]; covA[p] = run > 0; if (covA[p]) cntA++; }
        run = 0;
        for (int q = 0; q < lb; q++) { run += dB[q]; covB[q] = run > 0; if (covB[q]) cntB++; }

        r.covA = 100.0 * cntA / la;
        r.covB = 100.0 * cntB / lb;
        r.similarity = 100.0 * (cntA + cntB) / (la + lb);
        r.lcsLen = lcsLen;
        r.lcsText = lcsPos >= 0 ? ta.substring(lcsPos, lcsPos + lcsLen) : "";
        r.label = classify(r.similarity);

        // matching passages (maximal covered runs in A), longest first
        List<int[]> runs = new ArrayList<int[]>();
        int start = -1;
        for (int p = 0; p <= la; p++) {
            boolean cvd = p < la && covA[p];
            if (cvd && start < 0) start = p;
            else if (!cvd && start >= 0) { runs.add(new int[]{start, p}); start = -1; }
        }
        Collections.sort(runs, (x, y) -> (y[1] - y[0]) - (x[1] - x[0]));
        for (int i = 0; i < runs.size() && i < 8; i++) {
            int[] rr = runs.get(i);
            r.passages.add("[" + (rr[1] - rr[0]) + " chars] \"" + snippet(ta.substring(rr[0], rr[1]).trim()) + "\"");
        }
        r.nanos = System.nanoTime() - t0;
        return r;
    }

    static String snippet(String s) {
        return s.length() <= 170 ? s : s.substring(0, 170) + "...";
    }

    /** Compare ALL pairs and return them ranked from most to least similar. */
    static List<PairResult> analyzeAll(List<Doc> docs, int minLen, IntConsumer progress) {
        List<PairResult> out = new ArrayList<PairResult>();
        int n = docs.size();
        int total = n * (n - 1) / 2, done = 0;
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                PairResult r = compare(docs.get(i), docs.get(j), minLen);
                r.ai = i;
                r.bi = j;
                out.add(r);
                done++;
                if (progress != null && total > 0) progress.accept(done * 100 / total);
            }
        }
        Collections.sort(out, (x, y) -> {
            int c = Double.compare(y.similarity, x.similarity);
            return c != 0 ? c : Integer.compare(y.lcsLen, x.lcsLen);
        });
        return out;
    }

    static String describe(PairResult r) {
        StringBuilder sb = new StringBuilder();
        sb.append("Pair            : ").append(r.a.name).append("   <->   ").append(r.b.name).append('\n');
        sb.append(String.format("Similarity      : %.2f %%   (%s)%n", r.similarity, r.label));
        sb.append(String.format("Coverage of A   : %.2f %%      Coverage of B : %.2f %%%n", r.covA, r.covB));
        sb.append("Longest common substring (").append(r.lcsLen).append(" chars):\n   \"")
          .append(snippet(r.lcsText.trim())).append("\"\n");
        sb.append("Matching passages (longest first, text taken from A):\n");
        if (r.passages.isEmpty()) sb.append("   (none at this minimum match length)\n");
        for (String p : r.passages) sb.append("   ").append(p).append('\n');
        sb.append(String.format("Suffix-array comparison time: %.2f ms%n", r.nanos / 1e6));
        return sb.toString();
    }

    // =========================================================================
    //  5. FILE HELPERS + DEMO DATA
    // =========================================================================

    static String readFile(File f) throws IOException {
        byte[] bytes = Files.readAllBytes(f.toPath());
        try {
            return StandardCharsets.UTF_8.newDecoder()
                    .onMalformedInput(CodingErrorAction.REPORT)
                    .onUnmappableCharacter(CodingErrorAction.REPORT)
                    .decode(java.nio.ByteBuffer.wrap(bytes)).toString();
        } catch (CharacterCodingException e) {
            return new String(bytes, StandardCharsets.ISO_8859_1);
        }
    }

    static List<Doc> demoDocs() {
        List<Doc> d = new ArrayList<Doc>();
        d.add(new Doc("Doc1_Original",
            "A suffix array is a sorted array of all suffixes of a string. It stores only the starting "
          + "positions of the suffixes, which makes it far more memory efficient than a suffix tree. Once the "
          + "suffix array and the LCP array are built, searching for a pattern or finding repeated substrings "
          + "can be done quickly using binary search and simple scans. Suffix arrays are widely used in text "
          + "indexing, bioinformatics, data compression and plagiarism detection systems."));
        d.add(new Doc("Doc2_EditedCopy",
            "A suffix array is a sorted array of all suffixes of a string. It stores only the starting "
          + "positions of the suffixes, which makes it much more memory efficient than a suffix tree. Once the "
          + "suffix array and the LCP array are built, searching for a pattern or finding repeated substrings "
          + "can be done quickly using binary search and linear scans. Suffix arrays are widely used in text "
          + "indexing, genome analysis and plagiarism detection systems."));
        d.add(new Doc("Doc3_PartlyRelated",
            "Plagiarism detection tools compare student submissions against each other. Suffix arrays are "
          + "widely used in text indexing, bioinformatics, data compression and plagiarism detection systems. "
          + "A good tool must also handle very large collections of documents, so speed and memory usage "
          + "matter a lot in practice."));
        d.add(new Doc("Doc4_Unrelated",
            "The monsoon season in Hyderabad usually begins in June and brings heavy rain to the city. "
          + "Students carry umbrellas to campus, roads get waterlogged and the lakes around the city fill up "
          + "quickly. By September the weather becomes pleasant and the evenings are cool."));
        return d;
    }

    // =========================================================================
    //  6. COMMAND-LINE MODE
    // =========================================================================

    static void runCli(List<Doc> docs, int minLen, boolean details) {
        if (docs.size() < 2) {
            System.out.println("Need at least 2 documents to compare.");
            return;
        }
        System.out.println("Documents loaded : " + docs.size() + "   (minimum match length = " + minLen + ")");
        for (int i = 0; i < docs.size(); i++)
            System.out.println("  [" + (i + 1) + "] " + docs.get(i));
        long t0 = System.nanoTime();
        List<PairResult> res = analyzeAll(docs, minLen, null);
        long ms = (System.nanoTime() - t0) / 1_000_000;

        System.out.println();
        System.out.printf("%-5s %-11s %-8s %-8s %-8s %-19s %s%n",
                "Rank", "Similarity", "Cov A", "Cov B", "LCS", "Classification", "Document pair");
        System.out.println(new String(new char[100]).replace('\0', '-'));
        int rank = 1;
        for (PairResult r : res) {
            System.out.printf("%-5d %-11s %-8s %-8s %-8d %-19s %s  <->  %s%n", rank++,
                    String.format("%.1f%%", r.similarity), String.format("%.1f%%", r.covA),
                    String.format("%.1f%%", r.covB), r.lcsLen, r.label, r.a.name, r.b.name);
        }
        System.out.println("\nTotal analysis time: " + ms + " ms for " + res.size() + " pair(s).");
        if (details) {
            System.out.println("\n================ DETAILS ================");
            for (PairResult r : res) System.out.println(describe(r));
        } else {
            System.out.println("(use --details to see the matching passages)");
        }
    }

    // =========================================================================
    //  7. SWING GUI
    // =========================================================================

    static final class GUI extends JFrame {
        final DefaultListModel<Doc> docModel = new DefaultListModel<Doc>();
        final JList<Doc> docList = new JList<Doc>(docModel);
        final JSpinner minLenSpinner = new JSpinner(new SpinnerNumberModel(20, 3, 500, 1));
        final JComboBox<Object> rankAgainst = new JComboBox<Object>();
        final JButton analyzeBtn = new JButton("Analyze");
        final JButton exportBtn = new JButton("Export CSV");
        final JLabel status = new JLabel(" Add at least 2 documents (or click 'Load Demo'), then press Analyze.");
        final JProgressBar progress = new JProgressBar(0, 100);

        final String[] RANK_COLS = {"Rank", "Document A", "Document B", "Similarity %",
                "Coverage A %", "Coverage B %", "Longest match", "Classification"};
        final DefaultTableModel rankModel = new DefaultTableModel(RANK_COLS, 0) {
            @Override public boolean isCellEditable(int r, int c) { return false; }
        };
        final JTable rankTable = new JTable(rankModel);
        final DefaultTableModel matrixModel = new DefaultTableModel() {
            @Override public boolean isCellEditable(int r, int c) { return false; }
        };
        final JTable matrixTable = new JTable(matrixModel);
        final JTextArea passagesArea = new JTextArea();

        List<PairResult> allResults = new ArrayList<PairResult>();
        List<PairResult> shown = new ArrayList<PairResult>();
        List<Doc> analyzedDocs = new ArrayList<Doc>();
        boolean updatingCombo = false;

        javax.swing.Timer reanalyzeTimer;
        Runnable liveRecompute = new Runnable() { public void run() { } };

        GUI() {
            super("Document Similarity Analyzer  |  Suffix Array + Similarity Metrics  |  DSA-3");
            setDefaultCloseOperation(EXIT_ON_CLOSE);
            setLayout(new BorderLayout(6, 6));

            add(buildDocPanel(), BorderLayout.WEST);
            add(buildTopBar(), BorderLayout.NORTH);

            JTabbedPane tabs = new JTabbedPane();
            tabs.addTab("Ranking", buildRankingTab());
            tabs.addTab("Similarity Matrix", new JScrollPane(matrixTable));
            tabs.addTab("Live Compare", buildLiveTab());
            tabs.addTab("How it works", buildAboutTab());
            add(tabs, BorderLayout.CENTER);

            JPanel south = new JPanel(new BorderLayout(8, 0));
            south.add(status, BorderLayout.CENTER);
            progress.setPreferredSize(new Dimension(200, 16));
            south.add(progress, BorderLayout.EAST);
            south.setBorder(BorderFactory.createEmptyBorder(2, 6, 4, 6));
            add(south, BorderLayout.SOUTH);

            rankTable.setRowHeight(24);
            matrixTable.setRowHeight(24);
            rankTable.setAutoCreateRowSorter(false);
            rankTable.getColumnModel().getColumn(7).setCellRenderer(new LabelRenderer());
            passagesArea.setEditable(false);
            passagesArea.setFont(new Font(Font.MONOSPACED, Font.PLAIN, 13));
            passagesArea.setLineWrap(true);
            passagesArea.setWrapStyleWord(true);
            passagesArea.setText("Select a row in the ranking table to see the matching passages.");

            rankTable.getSelectionModel().addListSelectionListener(e -> {
                if (e.getValueIsAdjusting()) return;
                int row = rankTable.getSelectedRow();
                if (row >= 0 && row < shown.size()) passagesArea.setText(describe(shown.get(row)));
                passagesArea.setCaretPosition(0);
            });

            analyzeBtn.addActionListener(e -> runAnalysis());
            exportBtn.addActionListener(e -> exportCsv());
            rankAgainst.addActionListener(e -> { if (!updatingCombo) refreshViews(); });

            // "real-time": changing the minimum match length re-analyses automatically
            reanalyzeTimer = new javax.swing.Timer(450, e -> { if (!allResults.isEmpty()) runAnalysis(); });
            reanalyzeTimer.setRepeats(false);
            minLenSpinner.addChangeListener(e -> { reanalyzeTimer.restart(); liveRecompute.run(); });

            refreshCombo();
            setSize(1200, 700);
            setMinimumSize(new Dimension(900, 520));
            setLocationRelativeTo(null);
        }

        // ---------------- panels ----------------

        JPanel buildDocPanel() {
            JPanel p = new JPanel(new BorderLayout(4, 4));
            p.setBorder(BorderFactory.createTitledBorder("Input documents"));
            p.setPreferredSize(new Dimension(280, 100));
            docList.setSelectionMode(ListSelectionModel.MULTIPLE_INTERVAL_SELECTION);
            p.add(new JScrollPane(docList), BorderLayout.CENTER);

            JPanel btns = new JPanel(new GridLayout(0, 2, 4, 4));
            JButton addFiles = new JButton("Add TXT files");
            JButton addText = new JButton("Paste text");
            JButton remove = new JButton("Remove");
            JButton clear = new JButton("Clear all");
            JButton demo = new JButton("Load Demo");
            addFiles.addActionListener(e -> addFiles());
            addText.addActionListener(e -> addPasted());
            remove.addActionListener(e -> {
                List<Doc> sel = docList.getSelectedValuesList();
                for (Doc d : sel) docModel.removeElement(d);
                docsChanged();
            });
            clear.addActionListener(e -> { docModel.clear(); docsChanged(); });
            demo.addActionListener(e -> {
                docModel.clear();
                for (Doc d : demoDocs()) docModel.addElement(d);
                docsChanged();
                runAnalysis();
            });
            btns.add(addFiles); btns.add(addText); btns.add(remove); btns.add(clear); btns.add(demo);
            p.add(btns, BorderLayout.SOUTH);
            return p;
        }

        JPanel buildTopBar() {
            JPanel p = new JPanel(new FlowLayout(FlowLayout.LEFT, 10, 6));
            p.add(new JLabel("Minimum match length (chars):"));
            minLenSpinner.setToolTipText("Common substrings shorter than this are ignored (noise filter)");
            p.add(minLenSpinner);
            p.add(new JLabel("Rank against:"));
            rankAgainst.setPreferredSize(new Dimension(260, 26));
            p.add(rankAgainst);
            p.add(analyzeBtn);
            p.add(exportBtn);
            return p;
        }

        JSplitPane buildRankingTab() {
            JSplitPane sp = new JSplitPane(JSplitPane.VERTICAL_SPLIT,
                    new JScrollPane(rankTable), new JScrollPane(passagesArea));
            sp.setResizeWeight(0.55);
            return sp;
        }

        JPanel buildLiveTab() {
            final JTextArea left = new JTextArea(10, 30), right = new JTextArea(10, 30);
            left.setLineWrap(true); left.setWrapStyleWord(true);
            right.setLineWrap(true); right.setWrapStyleWord(true);
            final JLabel out = new JLabel("  Type or paste text in both boxes - the score updates as you type.");
            out.setFont(out.getFont().deriveFont(Font.BOLD, 15f));

            final javax.swing.Timer t = new javax.swing.Timer(300, null);
            t.setRepeats(false);
            final Runnable compute = new Runnable() {
                public void run() {
                    if (left.getText().trim().isEmpty() || right.getText().trim().isEmpty()) {
                        out.setText("  Type or paste text in both boxes - the score updates as you type.");
                        return;
                    }
                    int min = (Integer) minLenSpinner.getValue();
                    PairResult r = compare(new Doc("Left", left.getText()), new Doc("Right", right.getText()), min);
                    out.setText(String.format("  Similarity: %.1f %%   |   %s   |   longest common substring: %d chars   |   %.2f ms",
                            r.similarity, r.label, r.lcsLen, r.nanos / 1e6));
                }
            };
            t.addActionListener(e -> compute.run());
            liveRecompute = new Runnable() { public void run() { t.restart(); } };
            DocumentListener dl = new DocumentListener() {
                public void insertUpdate(DocumentEvent e) { t.restart(); }
                public void removeUpdate(DocumentEvent e) { t.restart(); }
                public void changedUpdate(DocumentEvent e) { t.restart(); }
            };
            left.getDocument().addDocumentListener(dl);
            right.getDocument().addDocumentListener(dl);

            JPanel boxes = new JPanel(new GridLayout(1, 2, 8, 0));
            JScrollPane l = new JScrollPane(left), r = new JScrollPane(right);
            l.setBorder(BorderFactory.createTitledBorder("Document 1"));
            r.setBorder(BorderFactory.createTitledBorder("Document 2"));
            boxes.add(l); boxes.add(r);
            JPanel p = new JPanel(new BorderLayout(4, 8));
            p.setBorder(BorderFactory.createEmptyBorder(6, 6, 6, 6));
            p.add(boxes, BorderLayout.CENTER);
            p.add(out, BorderLayout.SOUTH);
            return p;
        }

        JScrollPane buildAboutTab() {
            JTextArea a = new JTextArea(
                "PIPELINE\n"
              + "  Input documents -> Preprocess -> Suffix Array + LCP -> Common substrings\n"
              + "  -> Similarity score -> Compare all pairs -> Rank -> Display\n\n"
              + "PREPROCESSING\n"
              + "  Lower-case, remove punctuation/symbols, collapse whitespace.\n\n"
              + "SUFFIX ARRAY\n"
              + "  For each pair A,B a suffix array is built over  A + '#' + B  (O(n log n)),\n"
              + "  then Kasai's algorithm builds the LCP array (O(n)).\n"
              + "  One pass over the suffix array finds, for every position, the longest\n"
              + "  substring that also occurs in the other document.\n\n"
              + "SIMILARITY METRIC\n"
              + "  Characters that belong to a common substring of at least the minimum match\n"
              + "  length are 'covered'.\n"
              + "     Similarity % = (covered chars of A + covered chars of B) / (|A| + |B|) x 100\n"
              + "     Coverage A % = share of A that is found in B (and vice versa)\n\n"
              + "CLASSIFICATION\n"
              + "     >= 95 %  Near-Duplicate      80-95 %  Highly Similar\n"
              + "     40-80 %  Moderately Similar  < 40 %   Low Similarity\n\n"
              + "TIPS\n"
              + "  * Raise the minimum match length to ignore short common phrases.\n"
              + "  * 'Rank against' ranks every other document against one chosen document.\n"
              + "  * Live Compare recalculates the score while you type.\n"
              + "  * Input formats: .txt files or pasted text (PDF/DOCX = future scope).\n");
            a.setEditable(false);
            a.setFont(new Font(Font.MONOSPACED, Font.PLAIN, 13));
            a.setMargin(new Insets(10, 12, 10, 12));
            return new JScrollPane(a);
        }

        // ---------------- actions ----------------

        void addFiles() {
            JFileChooser fc = new JFileChooser();
            fc.setMultiSelectionEnabled(true);
            fc.setFileFilter(new FileNameExtensionFilter("Text files (*.txt)", "txt"));
            if (fc.showOpenDialog(this) != JFileChooser.APPROVE_OPTION) return;
            for (File f : fc.getSelectedFiles()) {
                try {
                    docModel.addElement(new Doc(f.getName(), readFile(f)));
                } catch (IOException ex) {
                    JOptionPane.showMessageDialog(this, "Could not read " + f.getName() + ": " + ex.getMessage(),
                            "Read error", JOptionPane.ERROR_MESSAGE);
                }
            }
            docsChanged();
        }

        void addPasted() {
            JTextField name = new JTextField("Pasted document " + (docModel.size() + 1));
            JTextArea ta = new JTextArea(12, 50);
            ta.setLineWrap(true);
            ta.setWrapStyleWord(true);
            JPanel p = new JPanel(new BorderLayout(4, 4));
            p.add(name, BorderLayout.NORTH);
            p.add(new JScrollPane(ta), BorderLayout.CENTER);
            int ok = JOptionPane.showConfirmDialog(this, p, "Add pasted text",
                    JOptionPane.OK_CANCEL_OPTION, JOptionPane.PLAIN_MESSAGE);
            if (ok == JOptionPane.OK_OPTION && !ta.getText().trim().isEmpty()) {
                docModel.addElement(new Doc(name.getText().trim().isEmpty() ? "Pasted" : name.getText().trim(), ta.getText()));
                docsChanged();
            }
        }

        void docsChanged() {
            allResults = new ArrayList<PairResult>();
            analyzedDocs = new ArrayList<Doc>();
            refreshCombo();
            refreshViews();
            status.setText(" " + docModel.size() + " document(s) loaded. Press Analyze.");
            progress.setValue(0);
        }

        void refreshCombo() {
            updatingCombo = true;
            rankAgainst.removeAllItems();
            rankAgainst.addItem("All pairs");
            for (int i = 0; i < docModel.size(); i++) rankAgainst.addItem(docModel.get(i));
            updatingCombo = false;
        }

        void runAnalysis() {
            if (docModel.size() < 2) {
                status.setText(" Please add at least 2 documents.");
                return;
            }
            final List<Doc> docs = Collections.list(docModel.elements());
            final int min = (Integer) minLenSpinner.getValue();
            analyzeBtn.setEnabled(false);
            progress.setValue(0);
            status.setText(" Analysing " + docs.size() + " documents ...");
            SwingWorker<List<PairResult>, Void> worker = new SwingWorker<List<PairResult>, Void>() {
                long t0;
                @Override protected List<PairResult> doInBackground() {
                    t0 = System.nanoTime();
                    return analyzeAll(docs, min, pct -> setProgress(pct));
                }
                @Override protected void done() {
                    try {
                        allResults = get();
                        analyzedDocs = docs;
                        refreshViews();
                        long ms = (System.nanoTime() - t0) / 1_000_000;
                        progress.setValue(100);
                        status.setText(" Done: " + allResults.size() + " pair(s) compared in " + ms
                                + " ms   (minimum match length = " + min + ")");
                        if (rankModel.getRowCount() > 0) rankTable.setRowSelectionInterval(0, 0);
                    } catch (Exception ex) {
                        status.setText(" Error: " + ex.getMessage());
                    } finally {
                        analyzeBtn.setEnabled(true);
                    }
                }
            };
            worker.addPropertyChangeListener(ev -> {
                if ("progress".equals(ev.getPropertyName())) progress.setValue((Integer) ev.getNewValue());
            });
            worker.execute();
        }

        void refreshViews() {
            rankModel.setRowCount(0);
            shown = new ArrayList<PairResult>();
            Object sel = rankAgainst.getSelectedItem();
            int rank = 1;
            for (PairResult r : allResults) {
                if (sel instanceof Doc && r.a != sel && r.b != sel) continue;
                shown.add(r);
                rankModel.addRow(new Object[]{rank++, r.a.name, r.b.name,
                        String.format("%.1f", r.similarity), String.format("%.1f", r.covA),
                        String.format("%.1f", r.covB), r.lcsLen, r.label});
            }
            // similarity matrix
            int n = analyzedDocs.size();
            Object[] cols = new Object[n + 1];
            cols[0] = "Document";
            for (int i = 0; i < n; i++) cols[i + 1] = analyzedDocs.get(i).name;
            Object[][] data = new Object[n][n + 1];
            for (int i = 0; i < n; i++) {
                data[i][0] = analyzedDocs.get(i).name;
                for (int j = 0; j < n; j++) data[i][j + 1] = (i == j) ? "100.0 %" : "-";
            }
            for (PairResult r : allResults) {
                String v = String.format("%.1f %%", r.similarity);
                data[r.ai][r.bi + 1] = v;
                data[r.bi][r.ai + 1] = v;
            }
            matrixModel.setDataVector(data, cols);
            if (rankModel.getRowCount() == 0)
                passagesArea.setText("Select a row in the ranking table to see the matching passages.");
        }

        void exportCsv() {
            if (allResults.isEmpty()) {
                status.setText(" Nothing to export yet - run Analyze first.");
                return;
            }
            JFileChooser fc = new JFileChooser();
            fc.setSelectedFile(new File("similarity_results.csv"));
            if (fc.showSaveDialog(this) != JFileChooser.APPROVE_OPTION) return;
            try (PrintWriter w = new PrintWriter(fc.getSelectedFile(), "UTF-8")) {
                w.println("Rank,Document A,Document B,Similarity %,Coverage A %,Coverage B %,Longest match (chars),Classification");
                int rank = 1;
                for (PairResult r : allResults)
                    w.printf("%d,\"%s\",\"%s\",%.2f,%.2f,%.2f,%d,%s%n", rank++, r.a.name, r.b.name,
                            r.similarity, r.covA, r.covB, r.lcsLen, r.label);
                status.setText(" Exported to " + fc.getSelectedFile().getAbsolutePath());
            } catch (IOException ex) {
                status.setText(" Export failed: " + ex.getMessage());
            }
        }
    }

    /** Colours the classification column. */
    static final class LabelRenderer extends DefaultTableCellRenderer {
        @Override
        public Component getTableCellRendererComponent(JTable t, Object v, boolean sel, boolean foc, int r, int c) {
            Component comp = super.getTableCellRendererComponent(t, v, sel, foc, r, c);
            if (!sel) {
                String s = String.valueOf(v);
                if (s.startsWith("Near"))          comp.setBackground(new Color(255, 190, 190));
                else if (s.startsWith("Highly"))   comp.setBackground(new Color(255, 215, 170));
                else if (s.startsWith("Moderate")) comp.setBackground(new Color(255, 245, 180));
                else                               comp.setBackground(new Color(200, 240, 200));
                comp.setForeground(Color.BLACK);
            }
            return comp;
        }
    }

    // =========================================================================
    //  8. MAIN
    // =========================================================================

    public static void main(String[] args) {
        int minLen = 20;
        boolean details = false, demo = false;
        List<File> files = new ArrayList<File>();
        for (String a : args) {
            if (a.startsWith("--min=")) minLen = Integer.parseInt(a.substring(6));
            else if (a.equals("--details")) details = true;
            else if (a.equals("--demo")) demo = true;
            else files.add(new File(a));
        }

        if (demo || !files.isEmpty() || GraphicsEnvironment.isHeadless()) {
            List<Doc> docs = new ArrayList<Doc>();
            if (demo || files.isEmpty()) docs = demoDocs();
            for (File f : files) {
                try {
                    docs.add(new Doc(f.getName(), readFile(f)));
                } catch (IOException e) {
                    System.out.println("Could not read " + f + " : " + e.getMessage());
                }
            }
            runCli(docs, minLen, details);
            return;
        }

        try {
            UIManager.setLookAndFeel(UIManager.getSystemLookAndFeelClassName());
        } catch (Exception ignored) { }
        final int min = minLen;
        SwingUtilities.invokeLater(() -> {
            GUI g = new GUI();
            g.minLenSpinner.setValue(min);
            g.setVisible(true);
        });
    }
}
