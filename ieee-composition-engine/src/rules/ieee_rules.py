"""
ieee_rules.py - THE IEEE RULEBOOK. Every IEEE rule the engine follows, in one place.

Each rule has:
    id       a short code you can quote, e.g. "LAY-02"
    group    what it is about (page layout, headings, references...)
    rule     the rule in plain English
    value    the exact value, when the rule has one (0.75 in, 24 pt, ...)
    source   which official IEEE document it comes from (see SOURCES)
    status   ENFORCED - the engine applies it automatically
             CHECKED  - the engine detects it and reports it (it cannot be
                        fixed safely by a program)
             AUTHOR   - written down for completeness; it needs the author's
                        judgement, so the engine leaves it alone
    where    the file that applies or checks it
    check    the compliance-score check that tests it (if any)

The VALUES (sizes, margins, labels, word lists) are defined in this file and
nowhere else. The rest of the engine reads them from here:

    render/ieee_style.py        page, columns, fonts, sizes, labels
    render/docx_renderer.py     headings, captions, references layout
    compose/ieee_text.py        wording, numbers and units, cross-references
    compose/references.py       IEEE reference format, venue abbreviations
    compose/citations.py        [n] citations in order of first citation
    compose/front_matter.py     abstract length, index terms, title, authors
    structure/ieee_mapper.py    IEEE section names and numbering
    evaluate/compliance.py      the compliance score, which tests these rules

Change a value here and the writer, the text rules and the checker all follow.

A readable table of every rule is generated from this file:
    python tools/export_rules.py      ->  docs/IEEE_RULES.md
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# ===========================================================================
# Where the rules come from
# ===========================================================================

SOURCES: Dict[str, Tuple[str, str]] = {
    "TEMPLATE": ("IEEE Manuscript Templates for Conference Proceedings",
                 "https://ieee.org/conferences/publishing/templates.html"),
    "GUIDE": ("IEEE conference template - author guidelines",
              "https://ewh.ieee.org/soc/ras/conf/financiallycosponsored/aim/2007/"
              "aim2007.ethz.ch/template/guidelines.pdf"),
    "LAYOUT": ("IEEE conference example paper - page layout and type sizes (Table I)",
               "https://ewh.ieee.org/conf/ius_2010/z_doc_misc/examplepaper.pdf"),
    "ESM": ("IEEE Editorial Style Manual for Authors",
            "https://journals.ieeeauthorcenter.ieee.org/your-role-in-article-production/"
            "ieee-editorial-style-manual/"),
    "REFGUIDE": ("IEEE Reference Guide",
                 "https://docs.google.com/document/d/e/2PACX-1vSBoV_hTIURPZKtQkxI6bAONJtjy"
                 "DLxQIUvCHi-XPzD8NKLcxRfZrpvQLFrjWZ3M6FFR_Obau3R-t63/pub"),
}

ENFORCED, CHECKED, AUTHOR = "enforced", "checked", "author"

# ===========================================================================
# VALUES - every number, label and word list the engine uses
# ===========================================================================

# --- Page and columns (LAY) -------------------------------------------------

@dataclass(frozen=True)
class PaperSize:
    name: str
    width_in: float
    height_in: float
    margin_top_in: float
    margin_bottom_in: float
    margin_side_in: float
    column_width_in: float
    column_gap_in: float
    first_line_indent_in: float


MM = 1 / 25.4      # one millimetre in inches

PAPER_SIZES: Dict[str, PaperSize] = {
    # 0.625 + 3.5 + 0.25 + 3.5 + 0.625 = 8.5 in - the page adds up exactly
    "letter": PaperSize("US Letter", 8.5, 11.0, 0.75, 1.0, 0.625, 3.5, 0.25, 0.2),
    "a4": PaperSize("A4", 210 * MM, 297 * MM, 19 * MM, 43 * MM, 13 * MM, 88 * MM, 4 * MM, 3.5 * MM),
}
DEFAULT_PAPER = "letter"
COLUMN_COUNT = 2

# --- Typography (TYP) ---------------------------------------------------------

FONT = "Times New Roman"
CODE_FONT = "Courier New"          # code listings only - not an IEEE rule, a readability choice

SIZE_PT: Dict[str, float] = {
    "title": 24,
    "author": 11,
    "affiliation": 10,
    "abstract": 9,          # abstract and index terms
    "body": 10,             # main text and equations
    "heading": 10,          # headings are NOT bigger: small capitals / italics instead
    "caption": 8,           # figure and table captions
    "reference": 8,
    "code": 7,
}

SPACING_PT: Dict[str, float] = {
    "heading1_before": 8, "heading1_after": 4,
    "heading2_before": 6, "heading2_after": 3,
    "caption_before": 4, "caption_after": 8,
    "title_after": 12, "author_block_after": 18,
}
REFERENCE_HANGING_INDENT_IN = 0.25
AUTHORS_PER_ROW = 3              # IEEE author block: up to three authors side by side

# --- Fixed wording ------------------------------------------------------------

LABELS: Dict[str, str] = {
    "abstract": "Abstract—",           # em dash, no space
    "index_terms": "Index Terms—",
    "references": "References",
    "acknowledgment": "Acknowledgment",  # IEEE spells it without the "e"
    "appendix": "Appendix",
    "figure": "Fig.",                    # always abbreviated
    "table": "TABLE",                    # capitals + Roman numeral
    "algorithm": "Algorithm",
    "section": "Section",                # "Section II-A", never "Subsection"
}

# --- Abstract and index terms (ABS) -------------------------------------------

ABSTRACT_WORDS = (150, 250)

# --- IEEE section skeleton (HDG) ----------------------------------------------
#  key -> (heading IEEE prints, usual order, numbered with a Roman numeral?)

SECTIONS: Dict[str, Tuple[str, int, bool]] = {
    "introduction":   ("Introduction", 1, True),
    "related_work":   ("Related Work", 2, True),
    "methodology":    ("Methodology", 3, True),
    "experiments":    ("Experimental Setup", 4, True),
    "results":        ("Results", 5, True),
    "discussion":     ("Discussion", 6, True),
    "conclusion":     ("Conclusion", 7, True),       # singular
    "acknowledgment": ("Acknowledgment", 8, False),  # never numbered
    "statement":      ("", 8, False),                # ethics / reproducibility statements
    "appendix":       ("Appendix", 9, False),
}
REQUIRED_SECTIONS = ["introduction", "methodology", "results", "conclusion"]

# --- References (REF) ---------------------------------------------------------

MAX_AUTHORS_BEFORE_ET_AL = 6       # more than six authors -> first author + "et al."
NO_DATE = "(n.d.)"

# Word abbreviations for journal and conference names (IEEE Reference Guide,
# "abbreviations of words in references"). Single-word names are never
# abbreviated, and small joining words are dropped once a name is abbreviated:
# "IEEE Transactions on Medical Imaging" -> "IEEE Trans. Med. Imag."
VENUE_ABBREVIATIONS: Dict[str, str] = {
    "Acoustics": "Acoust.", "Administration": "Admin.", "Advanced": "Adv.", "Advances": "Adv.",
    "Aeronautics": "Aeronaut.", "American": "Amer.", "Analysis": "Anal.", "Annals": "Ann.",
    "Annual": "Annu.", "Applications": "Appl.", "Applied": "Appl.", "Artificial": "Artif.",
    "Association": "Assoc.", "Automatic": "Autom.", "Automation": "Autom.",
    "Biomedical": "Biomed.", "Biology": "Biol.", "Bulletin": "Bull.", "Business": "Bus.",
    "Chemistry": "Chem.", "Communications": "Commun.", "Computational": "Comput.",
    "Computer": "Comput.", "Computers": "Comput.", "Computing": "Comput.",
    "Conference": "Conf.", "Cybernetics": "Cybern.", "Department": "Dept.", "Design": "Des.",
    "Development": "Develop.", "Digest": "Dig.", "Discovery": "Discov.", "Economics": "Econ.",
    "Electrical": "Elect.", "Electronic": "Electron.", "Electronics": "Electron.",
    "Empirical": "Empir.", "Engineering": "Eng.", "Environmental": "Environ.",
    "Evolutionary": "Evol.", "Foundation": "Found.", "Geoscience": "Geosci.",
    "Graphics": "Graph.", "Imaging": "Imag.", "Industrial": "Ind.", "Informatics": "Inform.",
    "Information": "Inf.", "Institute": "Inst.", "Intelligence": "Intell.",
    "Intelligent": "Intell.", "International": "Int.", "Journal": "J.", "Knowledge": "Knowl.",
    "Language": "Lang.", "Learning": "Learn.", "Letters": "Lett.", "Machine": "Mach.",
    "Magazine": "Mag.", "Management": "Manage.", "Materials": "Mater.",
    "Mathematical": "Math.", "Mathematics": "Math.", "Mechanical": "Mech.",
    "Medical": "Med.", "Medicine": "Med.", "National": "Nat.", "Natural": "Nat.",
    "Network": "Netw.", "Networks": "Netw.", "Newsletter": "Newslett.", "Nuclear": "Nucl.",
    "Optics": "Opt.", "Organization": "Org.", "Physics": "Phys.", "Physical": "Phys.",
    "Proceedings": "Proc.", "Processing": "Process.", "Production": "Prod.",
    "Quarterly": "Quart.", "Recognition": "Recognit.", "Record": "Rec.", "Report": "Rep.",
    "Representations": "Represent.", "Research": "Res.", "Review": "Rev.",
    "Robotics": "Robot.", "Science": "Sci.", "Sciences": "Sci.", "Security": "Secur.",
    "Selected": "Sel.", "Sensing": "Sens.", "Society": "Soc.", "Software": "Softw.",
    "Statistics": "Stat.", "Studies": "Stud.", "Symposium": "Symp.", "Systems": "Syst.",
    "Technical": "Tech.", "Technology": "Technol.", "Telecommunications": "Telecommun.",
    "Transactions": "Trans.", "Transportation": "Transp.", "Vehicular": "Veh.",
    "Vision": "Vis.", "Visualization": "Vis.",
}
VENUE_DROP_WORDS = {"of", "on", "the", "and", "for", "in", "&"}

# --- Citations (CIT) ----------------------------------------------------------

# words before a full stop that are abbreviations, not the end of a sentence:
# "et al. [3]" must stay as it is
ABBREVIATIONS_BEFORE_CITATION = {"al", "fig", "figs", "eq", "eqs", "vs", "etc", "e.g", "i.e",
                                 "no", "vol", "pp", "ref", "refs", "sec", "approx", "cf"}

# --- Numbers and units (NUM) --------------------------------------------------

# Units that get a space after the number ("10 mm"). Single letters (m, s, g,
# V, A) are left out on purpose: "5s" or "10M" is too often not a unit.
UNITS: List[str] = [
    "nm", "µm", "μm", "mm", "cm", "km", "mg", "kg", "ns", "µs", "μs", "ms",
    "Hz", "kHz", "MHz", "GHz", "THz", "mV", "kV", "mA", "kW", "MW", "GW", "mW",
    "dB", "dBm", "kB", "KB", "MB", "GB", "TB", "Kbps", "Mbps", "Gbps", "ppm", "°C", "°F",
]

# --- Writing style (STY) ------------------------------------------------------

CONTRACTIONS: Dict[str, str] = {
    "don't": "do not", "doesn't": "does not", "didn't": "did not", "can't": "cannot",
    "won't": "will not", "isn't": "is not", "aren't": "are not", "wasn't": "was not",
    "weren't": "were not", "it's": "it is", "that's": "that is", "there's": "there is",
    "we're": "we are", "they're": "they are", "we've": "we have", "couldn't": "could not",
    "shouldn't": "should not", "wouldn't": "would not", "hasn't": "has not",
    "haven't": "have not", "let's": "let us", "i'm": "I am", "you're": "you are",
    "we'll": "we will", "they've": "they have", "here's": "here is", "what's": "what is",
}

# Capitalised words that are NOT acronyms needing a definition: Roman
# numerals, IEEE's own labels, and units.
NOT_ACRONYMS = {"I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII",
                "TABLE", "IEEE", "OK", "USA", "UK", "EU"} | set(UNITS)

# Words kept in capitals when an ALL-CAPS title or heading is re-cased.
KEEP_UPPER = {"ai", "ml", "llm", "llms", "rl", "nlp", "cnn", "cnns", "rnn", "gpu", "cpu", "api",
              "iot", "ieee", "ct", "mri", "rag", "ui", "ux", "sql", "url", "gpt", "bert", "svm",
              "knn", "3d", "2d", "uav", "gan", "gans", "lstm", "dnn", "vr", "ar"}


# ===========================================================================
# THE RULES
# ===========================================================================

@dataclass(frozen=True)
class Rule:
    id: str
    group: str
    rule: str
    source: str                 # key of SOURCES
    status: str                 # ENFORCED / CHECKED / AUTHOR
    where: str = ""
    check: str = ""             # compliance check key, "" if none
    value: str = ""
    section: str = ""           # the part of the source document


_L = PAPER_SIZES["letter"]
_A = PAPER_SIZES["a4"]
R = "src/render/docx_renderer.py"
ST = "src/render/ieee_style.py"
TX = "src/compose/ieee_text.py"
RF = "src/compose/references.py"
CI = "src/compose/citations.py"
FM = "src/compose/front_matter.py"
MP = "src/structure/ieee_mapper.py"
PL = "src/placement/contextual.py"
CO = "src/evaluate/compliance.py"

RULES: List[Rule] = [
    # ---------------------------------------------------------------- page layout
    Rule("LAY-01", "Page layout", "Paper size is US Letter; A4 is available as an option.",
         "GUIDE", ENFORCED, ST, "page_size", f"{_L.width_in} x {_L.height_in} in (A4: 210 x 297 mm)",
         "Page layout"),
    Rule("LAY-02", "Page layout", "Page margins.",
         "LAYOUT", ENFORCED, ST, "margins",
         f"top {_L.margin_top_in} in, bottom {_L.margin_bottom_in} in, left/right "
         f"{_L.margin_side_in} in (A4: 19 / 43 / 13 mm)", "Page layout"),
    Rule("LAY-03", "Page layout", "The body is set in two columns.",
         "GUIDE", ENFORCED, R, "columns", f"{COLUMN_COUNT} columns, {_L.column_width_in} in wide "
         "(A4: 88 mm)", "Page layout"),
    Rule("LAY-04", "Page layout", "Gap between the two columns.",
         "LAYOUT", ENFORCED, ST, "columns", f"{_L.column_gap_in} in (A4: 4 mm)", "Page layout"),
    Rule("LAY-05", "Page layout", "Title and author block run across the full page width, "
         "above the two columns.", "TEMPLATE", ENFORCED, R),
    Rule("LAY-06", "Page layout", "Columns are justified (straight left and right edges).",
         "LAYOUT", ENFORCED, R),
    Rule("LAY-07", "Page layout", "On the last page, the two columns end at the same height.",
         "LAYOUT", ENFORCED, R),
    Rule("LAY-08", "Page layout", "No page numbers, headers or footers are added.",
         "LAYOUT", ENFORCED, R),
    Rule("LAY-09", "Page layout", "Do not shrink fonts or spacing to fit; keep within the "
         "conference page limit.", "GUIDE", AUTHOR),

    # ---------------------------------------------------------------- typography
    Rule("TYP-01", "Typography", "Times New Roman throughout.", "LAYOUT", ENFORCED, ST, "font", FONT),
    Rule("TYP-02", "Typography", "Paper title size.", "LAYOUT", ENFORCED, ST, "title_size",
         f"{SIZE_PT['title']:g} pt"),
    Rule("TYP-03", "Typography", "Author names size.", "LAYOUT", ENFORCED, ST, "",
         f"{SIZE_PT['author']:g} pt"),
    Rule("TYP-04", "Typography", "Affiliations size (italic).", "LAYOUT", ENFORCED, ST, "",
         f"{SIZE_PT['affiliation']:g} pt"),
    Rule("TYP-05", "Typography", "Main text and equations size.", "LAYOUT", ENFORCED, ST,
         "body_size", f"{SIZE_PT['body']:g} pt"),
    Rule("TYP-06", "Typography", "Abstract and index terms size (bold).", "LAYOUT", ENFORCED, ST,
         "", f"{SIZE_PT['abstract']:g} pt"),
    Rule("TYP-07", "Typography", "Section headings are not enlarged: small capitals (level 1) "
         "or italics (levels 2-4).", "LAYOUT", ENFORCED, R, "", f"{SIZE_PT['heading']:g} pt"),
    Rule("TYP-08", "Typography", "Figure and table caption size.", "LAYOUT", ENFORCED, ST, "",
         f"{SIZE_PT['caption']:g} pt"),
    Rule("TYP-09", "Typography", "Reference list size.", "LAYOUT", ENFORCED, ST, "",
         f"{SIZE_PT['reference']:g} pt"),
    Rule("TYP-10", "Typography", "Emphasis uses italics, never underlining.", "GUIDE", ENFORCED, R),
    Rule("TYP-11", "Typography", "Body paragraphs start with a first-line indent.",
         "TEMPLATE", ENFORCED, ST, "", f"{_L.first_line_indent_in} in (A4: 3.5 mm)"),

    # ---------------------------------------------------------------- title and authors
    Rule("TTL-01", "Title and authors", "The title is in title case, not all capitals.",
         "GUIDE", ENFORCED, FM, "title_case", "", "Title"),
    Rule("TTL-02", "Title and authors", "\"(Invited)\" is not written in the title.",
         "GUIDE", ENFORCED, FM, "title_case", "", "Title"),
    Rule("TTL-03", "Title and authors", "A space goes between author initials: \"J. K. Smith\".",
         "GUIDE", ENFORCED, FM, "", "", "Author names"),
    Rule("TTL-06", "Title and authors", "Each author is listed separately, in a column of their "
         "own with their affiliation(s) underneath; up to three authors per row (the IEEE author "
         "block). Superscript affiliation numbers are replaced by the affiliation itself.",
         "TEMPLATE", ENFORCED, R, "author_block", f"{AUTHORS_PER_ROW} per row", "Authors"),
    Rule("TTL-07", "Title and authors", "Figure captions and text inside a front-page figure are "
         "never taken as authors or affiliations.", "TEMPLATE", ENFORCED,
         "src/structure/classifier.py", "author_block"),
    Rule("TTL-04", "Title and authors", "Avoid abbreviations and long formulas in the title.",
         "GUIDE", AUTHOR, "", "", "", "Title"),
    Rule("TTL-05", "Title and authors", "Each author gives affiliation, city, country and email.",
         "TEMPLATE", AUTHOR),

    # ---------------------------------------------------------------- abstract and index terms
    Rule("ABS-01", "Abstract and index terms", "The abstract starts with the run-in label "
         "\"Abstract—\" in bold italic.", "TEMPLATE", ENFORCED, R, "abstract_label",
         LABELS["abstract"]),
    Rule("ABS-02", "Abstract and index terms", "The abstract is a single paragraph.",
         "ESM", ENFORCED, R, "", "", "Abstract"),
    Rule("ABS-03", "Abstract and index terms", "The abstract is 150 to 250 words. Over 250 is "
         "shortened (Gemini, checked); under 250 is reported, never padded.", "ESM", CHECKED, FM,
         "abstract_length", f"{ABSTRACT_WORDS[0]}-{ABSTRACT_WORDS[1]} words", "Abstract"),
    Rule("ABS-04", "Abstract and index terms", "The abstract contains no citations.",
         "ESM", ENFORCED, FM, "abstract_clean", "", "Abstract"),
    Rule("ABS-05", "Abstract and index terms", "The abstract contains no numbered equations or "
         "footnotes.", "ESM", AUTHOR, "", "", "", "Abstract"),
    Rule("ABS-06", "Abstract and index terms", "Index terms follow the abstract with the run-in "
         "label \"Index Terms—\" in bold italic.", "ESM", ENFORCED, R, "index_terms",
         LABELS["index_terms"], "Index Terms"),
    Rule("ABS-07", "Abstract and index terms", "Index terms are in alphabetical order.",
         "ESM", ENFORCED, FM, "index_terms", "", "Index Terms"),
    Rule("ABS-08", "Abstract and index terms", "Only the first word of the index-terms list is "
         "capitalised (plus acronyms and proper nouns).", "ESM", ENFORCED, R, "", "", "Index Terms"),

    # ---------------------------------------------------------------- headings
    Rule("HDG-01", "Headings", "Level 1: Roman numeral, small capitals, centred - "
         "\"I. INTRODUCTION\".", "ESM", ENFORCED, MP, "heading_numbering", "", "Section headings"),
    Rule("HDG-02", "Headings", "Level 2: capital letter, italic, flush left - \"A. Dataset\".",
         "ESM", ENFORCED, R, "", "", "Section headings"),
    Rule("HDG-03", "Headings", "Level 3: \"1)\" in italics, run into the paragraph and "
         "followed by a colon - \"1) Training: text...\".", "ESM", ENFORCED, R, "", "",
         "Section headings"),
    Rule("HDG-04", "Headings", "Level 4: \"a)\" in italics, indented, run into the paragraph.",
         "ESM", ENFORCED, R, "", "", "Section headings"),
    Rule("HDG-05", "Headings", "Sections carry the standard IEEE names (Introduction ... "
         "Conclusion).", "TEMPLATE", ENFORCED, MP, "heading_names",
         ", ".join(SECTIONS[k][0] for k in REQUIRED_SECTIONS)),
    Rule("HDG-06", "Headings", "Acknowledgment and References are never numbered.",
         "ESM", ENFORCED, MP, "", "", "Section headings"),
    Rule("HDG-07", "Headings", "\"Introduction\", \"Conclusion\" and \"Acknowledgment\" are "
         "singular.", "ESM", ENFORCED, MP, "", "", "Section headings"),
    Rule("HDG-08", "Headings", "One appendix is just \"Appendix\"; several are \"Appendix A\", "
         "\"Appendix B\".", "ESM", ENFORCED, MP, "", "", "Appendixes"),
    Rule("HDG-09", "Headings", "Cross-references say \"Section II\" or \"Section II-A\", never "
         "\"Subsection\"; the author's numbers are converted to the IEEE numbers.", "ESM",
         ENFORCED, TX, "", "", "Section headings"),

    # ---------------------------------------------------------------- figures
    Rule("FIG-01", "Figures", "The figure caption goes BELOW the figure, centred.",
         "GUIDE", ENFORCED, R, "caption_style", "", "Figures"),
    Rule("FIG-02", "Figures", "The caption starts \"Fig. 1.\".", "ESM", ENFORCED, R,
         "caption_style", LABELS["figure"], "Figure captions"),
    Rule("FIG-03", "Figures", "In the text a figure is always \"Fig. 1\", even at the start of "
         "a sentence.", "ESM", ENFORCED, TX, "visual_mentions", "", "Figure captions"),
    Rule("FIG-04", "Figures", "Parts of a figure are cited as \"Fig. 1(a)\".", "ESM", ENFORCED, TX,
         "visual_mentions", "", "Figure captions"),
    Rule("FIG-05", "Figures", "Each figure and table is placed after the paragraph that first "
         "mentions it (the engine's novelty).", "GUIDE", ENFORCED, PL, "placement"),
    Rule("FIG-06", "Figures", "Figures and tables are first mentioned in number order.",
         "ESM", CHECKED, CO, "visual_order", "", "Figure captions"),
    Rule("FIG-07", "Figures", "A figure is never wider than its column.", "GUIDE", ENFORCED, R),
    Rule("FIG-08", "Figures", "No caption, text box or border inside the figure image.",
         "GUIDE", AUTHOR, "", "", "", "Figures"),
    Rule("FIG-09", "Figures", "Axis labels use words with units: \"Magnetization (kA/m)\".",
         "GUIDE", AUTHOR, "", "", "", "Figures"),
    Rule("FIG-10", "Figures", "A reused figure cites its source at the end of the caption.",
         "ESM", AUTHOR, "", "", "", "Figure captions"),

    # ---------------------------------------------------------------- tables
    Rule("TAB-01", "Tables", "The table title goes ABOVE the table.", "GUIDE", ENFORCED, R,
         "caption_style", "", "Tables"),
    Rule("TAB-02", "Tables", "Tables are labelled \"TABLE I\" with a Roman numeral, centred.",
         "ESM", ENFORCED, R, "caption_style", LABELS["table"], "Table captions"),
    Rule("TAB-03", "Tables", "The table title is in small capitals with no full stop at the end.",
         "ESM", ENFORCED, R, "", "", "Table captions"),
    Rule("TAB-04", "Tables", "In the text a table is \"Table I\" (written in full, Roman "
         "numeral).", "GUIDE", ENFORCED, TX, "visual_mentions", "", "Tables"),
    Rule("TAB-05", "Tables", "Tables use horizontal lines only.", "GUIDE", AUTHOR, "", "", "",
         "Tables"),
    Rule("TAB-06", "Tables", "Table footnotes use letters (a, b, c).", "GUIDE", AUTHOR, "", "", "",
         "Tables"),

    # ---------------------------------------------------------------- equations
    Rule("EQN-01", "Equations", "Equation numbers are in brackets, flush right: \"(1)\".",
         "ESM", ENFORCED, R, "", "", "Equations"),
    Rule("EQN-02", "Equations", "In the text an equation is \"(1)\"; \"Equation (1)\" only at "
         "the start of a sentence.", "ESM", ENFORCED, TX, "", "", "Equations"),
    Rule("EQN-03", "Equations", "Equations are numbered consecutively, with no gaps or repeats.",
         "ESM", AUTHOR, "", "", "", "Equations"),
    Rule("EQN-04", "Equations", "Every symbol is defined; variables are in italics; an equation "
         "ending a sentence takes a full stop.", "GUIDE", AUTHOR, "", "", "", "Equations"),

    # ---------------------------------------------------------------- citations
    Rule("CIT-01", "Citations", "Citations are numbers in square brackets: [1].",
         "REFGUIDE", ENFORCED, CI, "citation_style", "", "Citing references"),
    Rule("CIT-02", "Citations", "References are numbered in the order they are first cited.",
         "GUIDE", ENFORCED, CI, "reference_order", "", "References"),
    Rule("CIT-03", "Citations", "Several citations are written out one by one: \"[3], [4], [5]\".",
         "REFGUIDE", ENFORCED, CI, "", "", "Citing references"),
    Rule("CIT-04", "Citations", "A citation sits inside the sentence punctuation: \"...in [3].\"",
         "REFGUIDE", ENFORCED, TX, "", "", "Citing references"),
    Rule("CIT-05", "Citations", "Write \"in [3]\", not \"in reference [3]\" or \"Ref. [3]\".",
         "GUIDE", ENFORCED, TX, "", "", "References"),
    Rule("CIT-06", "Citations", "Name three or more authors in the text as \"Smith et al. [4]\".",
         "REFGUIDE", AUTHOR, "", "", "", "Citing references"),

    # ---------------------------------------------------------------- references
    Rule("REF-01", "References", "The list is numbered \"[1]\" flush left, with a hanging indent.",
         "REFGUIDE", ENFORCED, R, "", "", "Reference list"),
    Rule("REF-02", "References", "Author initials come before the surname: \"J. K. Author\".",
         "REFGUIDE", ENFORCED, RF, "reference_style", "", "Author names"),
    Rule("REF-03", "References", "More than six authors: the first author followed by \"et al.\"",
         "REFGUIDE", ENFORCED, RF, "", str(MAX_AUTHORS_BEFORE_ET_AL), "Author names"),
    Rule("REF-04", "References", "The article title is in double quotes, comma inside: "
         "\"Title,\"", "REFGUIDE", ENFORCED, RF, "reference_style", "", "Titles"),
    Rule("REF-05", "References", "Journal, conference and book names are in italics.",
         "REFGUIDE", ENFORCED, RF, "", "", "Titles"),
    Rule("REF-06", "References", "Journal and conference names are abbreviated with IEEE's word "
         "abbreviations (\"Transactions\" -> \"Trans.\").", "REFGUIDE", ENFORCED, RF,
         "venue_abbreviation", f"{len(VENUE_ABBREVIATIONS)} word abbreviations", "Abbreviations"),
    Rule("REF-07", "References", "Conference papers read \"in Proc. ...\".", "REFGUIDE", ENFORCED,
         RF, "", "", "Conference papers"),
    Rule("REF-08", "References", "Volume, issue and pages as \"vol. 3, no. 2, pp. 10–20\" with an "
         "en dash.", "REFGUIDE", ENFORCED, RF, "", "", "Periodicals"),
    Rule("REF-09", "References", "arXiv preprints: J. K. Author, \"Title,\" year, arXiv:xxxx.",
         "REFGUIDE", ENFORCED, RF, "", "", "Preprints"),
    Rule("REF-10", "References", "Every reference has a year; \"(n.d.)\" when there is none.",
         "REFGUIDE", ENFORCED, RF, "", NO_DATE, "Dates"),
    Rule("REF-11", "References", "Every reference ends with a full stop, except one ending in a "
         "URL.", "REFGUIDE", ENFORCED, RF, "", "", "Punctuation"),
    Rule("REF-12", "References", "A DOI is written \"doi: 10.xxxx.\"; an online source as "
         "\"[Online]. Available: URL\".", "REFGUIDE", ENFORCED, RF, "", "", "Online sources"),

    # ---------------------------------------------------------------- numbers and units
    Rule("NUM-01", "Numbers and units", "A decimal has a zero before the point: \"0.25\", not "
         "\".25\".", "ESM", ENFORCED, TX, "number_style", "", "Numbers"),
    Rule("NUM-02", "Numbers and units", "A space goes between a number and its unit: \"10 mm\".",
         "ESM", ENFORCED, TX, "number_style", f"{len(UNITS)} units", "Units"),
    Rule("NUM-03", "Numbers and units", "No space before the percent sign: \"2%\".",
         "ESM", ENFORCED, TX, "number_style", "", "Numbers"),
    Rule("NUM-04", "Numbers and units", "Ranges use an en dash; percentages repeat the sign: "
         "\"40–50 mm\", \"20%–30%\".", "ESM", ENFORCED, TX, "number_style", "", "Numbers"),
    Rule("NUM-05", "Numbers and units", "Numbers below 11 are spelled out in ordinary text.",
         "ESM", AUTHOR, "", "", "", "Numbers"),
    Rule("NUM-06", "Numbers and units", "SI units are used; SI and CGS are not mixed.",
         "GUIDE", AUTHOR, "", "", "", "Units"),
    Rule("NUM-07", "Numbers and units", "Dates are written day month year: \"4 June 2002\".",
         "ESM", AUTHOR, "", "", "", "Numbers"),

    # ---------------------------------------------------------------- writing style
    Rule("STY-01", "Writing style", "No contractions: \"do not\", not \"don't\".", "GUIDE",
         ENFORCED, TX, "", f"{len(CONTRACTIONS)} contractions", "Other recommendations"),
    Rule("STY-02", "Writing style", "Every acronym is defined at its first use: \"convolutional "
         "neural network (CNN)\".", "ESM", CHECKED, CO, "acronyms", "", "Abbreviations and acronyms"),
    Rule("STY-03", "Writing style", "One space after a full stop.", "GUIDE", ENFORCED, TX),
    Rule("STY-04", "Writing style", "Use the serial comma: \"A, B, and C\".", "GUIDE", AUTHOR),
    Rule("STY-05", "Writing style", "American spelling (Merriam-Webster).", "ESM", AUTHOR),

    # ---------------------------------------------------------------- back matter
    Rule("ACK-01", "Back matter", "The heading is spelled \"Acknowledgment\" (no \"e\").",
         "ESM", ENFORCED, MP, "", LABELS["acknowledgment"], "Acknowledgment"),
    Rule("ACK-02", "Back matter", "Order at the end: appendices, then Acknowledgment, then "
         "References.", "ESM", ENFORCED, R, "", "", "Acknowledgment"),
    Rule("ACK-03", "Back matter", "The acknowledgment is in the third person; funding goes in "
         "the first-page footnote.", "ESM", AUTHOR, "", "", "", "Acknowledgment"),
]

RULES_BY_ID: Dict[str, Rule] = {r.id: r for r in RULES}


def rule(rule_id: str) -> Rule:
    """Look up one rule: rule("LAY-02")."""
    return RULES_BY_ID[rule_id]


def rules_for_check(check_key: str) -> List[str]:
    """The rule IDs a compliance check tests: rules_for_check("margins") -> ["LAY-02"]."""
    return [r.id for r in RULES if r.check == check_key]


def summary() -> Dict[str, int]:
    out = {ENFORCED: 0, CHECKED: 0, AUTHOR: 0}
    for r in RULES:
        out[r.status] += 1
    return out


def source_name(r: Rule) -> str:
    name = SOURCES[r.source][0]
    return f"{name} - {r.section}" if r.section else name


__all__ = ["RULES", "RULES_BY_ID", "Rule", "rule", "rules_for_check", "summary", "SOURCES",
           "ENFORCED", "CHECKED", "AUTHOR", "PAPER_SIZES", "DEFAULT_PAPER", "PaperSize"]
