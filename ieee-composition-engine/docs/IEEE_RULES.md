# IEEE Rulebook

Every IEEE formatting rule the IEEE Manuscript Composition Engine follows.
This file is generated from `src/rules/ieee_rules.py` by `python tools/export_rules.py`;
edit the rulebook, not this file.

**99 rules:** 78 applied automatically, 3 checked and reported, 18 left to the author's judgement.

| Status | Meaning |
|---|---|
| Applied automatically | The engine changes the manuscript to follow the rule. |
| Checked and reported | The engine measures the rule and reports it, but a program cannot fix it safely. |
| Author's judgement | Written down for completeness; it needs a human decision. |

## Sources

- **TEMPLATE**: [IEEE Manuscript Templates for Conference Proceedings](https://ieee.org/conferences/publishing/templates.html)
- **GUIDE**: [IEEE conference template - author guidelines](https://ewh.ieee.org/soc/ras/conf/financiallycosponsored/aim/2007/aim2007.ethz.ch/template/guidelines.pdf)
- **LAYOUT**: [IEEE conference example paper - page layout and type sizes (Table I)](https://ewh.ieee.org/conf/ius_2010/z_doc_misc/examplepaper.pdf)
- **ESM**: [IEEE Editorial Style Manual for Authors](https://journals.ieeeauthorcenter.ieee.org/your-role-in-article-production/ieee-editorial-style-manual/)
- **REFGUIDE**: [IEEE Reference Guide](https://docs.google.com/document/d/e/2PACX-1vSBoV_hTIURPZKtQkxI6bAONJtjyDLxQIUvCHi-XPzD8NKLcxRfZrpvQLFrjWZ3M6FFR_Obau3R-t63/pub)

## Page layout

| ID | Rule | Value | Status | Source | Applied in | Score check |
|---|---|---|---|---|---|---|
| LAY-01 | Paper size is US Letter; A4 is available as an option. | 8.5 x 11.0 in (A4: 210 x 297 mm) | Applied automatically | GUIDE: Page layout | `src/render/ieee_style.py` | `page_size` |
| LAY-02 | Page margins. | top 0.75 in, bottom 1.0 in, left/right 0.625 in (A4: 19 / 43 / 13 mm) | Applied automatically | LAYOUT: Page layout | `src/render/ieee_style.py` | `margins` |
| LAY-03 | The body is set in two columns. | 2 columns, 3.5 in wide (A4: 88 mm) | Applied automatically | GUIDE: Page layout | `src/render/docx_renderer.py` | `columns` |
| LAY-04 | Gap between the two columns. | 0.25 in (A4: 4 mm) | Applied automatically | LAYOUT: Page layout | `src/render/ieee_style.py` | `columns` |
| LAY-05 | Title and author block run across the full page width, above the two columns. |  | Applied automatically | TEMPLATE | `src/render/docx_renderer.py` |  |
| LAY-06 | Columns are justified (straight left and right edges). |  | Applied automatically | LAYOUT | `src/render/docx_renderer.py` |  |
| LAY-07 | On the last page, the two columns end at the same height. |  | Applied automatically | LAYOUT | `src/render/docx_renderer.py` |  |
| LAY-08 | No page numbers, headers or footers are added. |  | Applied automatically | LAYOUT | `src/render/docx_renderer.py` |  |
| LAY-09 | Do not shrink fonts or spacing to fit; keep within the conference page limit. |  | Author's judgement | GUIDE |  |  |

## Typography

| ID | Rule | Value | Status | Source | Applied in | Score check |
|---|---|---|---|---|---|---|
| TYP-01 | Times New Roman throughout. | Times New Roman | Applied automatically | LAYOUT | `src/render/ieee_style.py` | `font` |
| TYP-02 | Paper title size. | 24 pt | Applied automatically | LAYOUT | `src/render/ieee_style.py` | `title_size` |
| TYP-03 | Author names size. | 11 pt | Applied automatically | LAYOUT | `src/render/ieee_style.py` |  |
| TYP-04 | Affiliations size (italic). | 10 pt | Applied automatically | LAYOUT | `src/render/ieee_style.py` |  |
| TYP-05 | Main text and equations size. | 10 pt | Applied automatically | LAYOUT | `src/render/ieee_style.py` | `body_size` |
| TYP-06 | Abstract and index terms size (bold). | 9 pt | Applied automatically | LAYOUT | `src/render/ieee_style.py` |  |
| TYP-07 | Section headings are not enlarged: small capitals (level 1) or italics (levels 2-4). | 10 pt | Applied automatically | LAYOUT | `src/render/docx_renderer.py` |  |
| TYP-08 | Figure and table caption size. | 8 pt | Applied automatically | LAYOUT | `src/render/ieee_style.py` |  |
| TYP-09 | Reference list size. | 8 pt | Applied automatically | LAYOUT | `src/render/ieee_style.py` |  |
| TYP-10 | Emphasis uses italics, never underlining. |  | Applied automatically | GUIDE | `src/render/docx_renderer.py` |  |
| TYP-11 | Body paragraphs start with a first-line indent. | 0.2 in (A4: 3.5 mm) | Applied automatically | TEMPLATE | `src/render/ieee_style.py` |  |

## Title and authors

| ID | Rule | Value | Status | Source | Applied in | Score check |
|---|---|---|---|---|---|---|
| TTL-01 | The title is in title case, not all capitals. |  | Applied automatically | GUIDE: Title | `src/compose/front_matter.py` | `title_case` |
| TTL-02 | "(Invited)" is not written in the title. |  | Applied automatically | GUIDE: Title | `src/compose/front_matter.py` | `title_case` |
| TTL-03 | A space goes between author initials: "J. K. Smith". |  | Applied automatically | GUIDE: Author names | `src/compose/front_matter.py` |  |
| TTL-06 | Each author is listed separately, in a column of their own with their affiliation(s) underneath; up to three authors per row (the IEEE author block). Superscript affiliation numbers are replaced by the affiliation itself. | 3 per row | Applied automatically | TEMPLATE: Authors | `src/render/docx_renderer.py` | `author_block` |
| TTL-07 | Figure captions and text inside a front-page figure are never taken as authors or affiliations. |  | Applied automatically | TEMPLATE | `src/structure/classifier.py` | `author_block` |
| TTL-04 | Avoid abbreviations and long formulas in the title. |  | Author's judgement | GUIDE: Title |  |  |
| TTL-05 | Each author gives affiliation, city, country and email. |  | Author's judgement | TEMPLATE |  |  |

## Abstract and index terms

| ID | Rule | Value | Status | Source | Applied in | Score check |
|---|---|---|---|---|---|---|
| ABS-01 | The abstract starts with the run-in label "Abstract—" in bold italic. | Abstract— | Applied automatically | TEMPLATE | `src/render/docx_renderer.py` | `abstract_label` |
| ABS-02 | The abstract is a single paragraph. |  | Applied automatically | ESM: Abstract | `src/render/docx_renderer.py` |  |
| ABS-03 | The abstract is 150 to 250 words. Over 250 is shortened (Gemini, checked); under 250 is reported, never padded. | 150-250 words | Checked and reported | ESM: Abstract | `src/compose/front_matter.py` | `abstract_length` |
| ABS-04 | The abstract contains no citations. |  | Applied automatically | ESM: Abstract | `src/compose/front_matter.py` | `abstract_clean` |
| ABS-05 | The abstract contains no numbered equations or footnotes. |  | Author's judgement | ESM: Abstract |  |  |
| ABS-06 | Index terms follow the abstract with the run-in label "Index Terms—" in bold italic. | Index Terms— | Applied automatically | ESM: Index Terms | `src/render/docx_renderer.py` | `index_terms` |
| ABS-07 | Index terms are in alphabetical order. |  | Applied automatically | ESM: Index Terms | `src/compose/front_matter.py` | `index_terms` |
| ABS-08 | Only the first word of the index-terms list is capitalised (plus acronyms and proper nouns). |  | Applied automatically | ESM: Index Terms | `src/render/docx_renderer.py` |  |

## Headings

| ID | Rule | Value | Status | Source | Applied in | Score check |
|---|---|---|---|---|---|---|
| HDG-01 | Level 1: Roman numeral, small capitals, centred - "I. INTRODUCTION". |  | Applied automatically | ESM: Section headings | `src/structure/ieee_mapper.py` | `heading_numbering` |
| HDG-02 | Level 2: capital letter, italic, flush left - "A. Dataset". |  | Applied automatically | ESM: Section headings | `src/render/docx_renderer.py` |  |
| HDG-03 | Level 3: "1)" in italics, run into the paragraph and followed by a colon - "1) Training: text...". |  | Applied automatically | ESM: Section headings | `src/render/docx_renderer.py` |  |
| HDG-04 | Level 4: "a)" in italics, indented, run into the paragraph. |  | Applied automatically | ESM: Section headings | `src/render/docx_renderer.py` |  |
| HDG-05 | Sections carry the standard IEEE names (Introduction ... Conclusion). | Introduction, Methodology, Results, Conclusion | Applied automatically | TEMPLATE | `src/structure/ieee_mapper.py` | `heading_names` |
| HDG-06 | Acknowledgment and References are never numbered. |  | Applied automatically | ESM: Section headings | `src/structure/ieee_mapper.py` |  |
| HDG-07 | "Introduction", "Conclusion" and "Acknowledgment" are singular. |  | Applied automatically | ESM: Section headings | `src/structure/ieee_mapper.py` |  |
| HDG-08 | One appendix is just "Appendix"; several are "Appendix A", "Appendix B". |  | Applied automatically | ESM: Appendixes | `src/structure/ieee_mapper.py` |  |
| HDG-09 | Cross-references say "Section II" or "Section II-A", never "Subsection"; the author's numbers are converted to the IEEE numbers. |  | Applied automatically | ESM: Section headings | `src/compose/ieee_text.py` |  |

## Figures

| ID | Rule | Value | Status | Source | Applied in | Score check |
|---|---|---|---|---|---|---|
| FIG-01 | The figure caption goes BELOW the figure, centred. |  | Applied automatically | GUIDE: Figures | `src/render/docx_renderer.py` | `caption_style` |
| FIG-02 | The caption starts "Fig. 1." (always abbreviated) followed by an em space, then the caption text, which ends with a full stop. | Fig. 1. Caption text. | Applied automatically | ESM: Figure captions | `src/render/docx_renderer.py` | `caption_style` |
| FIG-03 | In the text a figure is always "Fig. 1", even at the start of a sentence. |  | Applied automatically | ESM: Figure captions | `src/compose/ieee_text.py` | `visual_mentions` |
| FIG-04 | Parts of a figure are cited as "Fig. 1(a)". |  | Applied automatically | ESM: Figure captions | `src/compose/ieee_text.py` | `visual_mentions` |
| FIG-05 | Each figure and table is placed after the paragraph that first mentions it (the engine's novelty). |  | Applied automatically | GUIDE | `src/placement/contextual.py` | `placement` |
| FIG-06 | Figures and tables are first mentioned in number order. |  | Checked and reported | ESM: Figure captions | `src/evaluate/compliance.py` | `visual_order` |
| FIG-07 | A figure is never wider than its column. |  | Applied automatically | GUIDE | `src/render/docx_renderer.py` |  |
| FIG-08 | No caption, text box or border inside the figure image. |  | Author's judgement | GUIDE: Figures |  |  |
| FIG-09 | Axis labels use words with units: "Magnetization (kA/m)". |  | Author's judgement | GUIDE: Figures |  |  |
| FIG-10 | A reused figure cites its source at the end of the caption. |  | Author's judgement | ESM: Figure captions |  |  |
| FIG-11 | Caption text starts with a capital letter and not with "A", "An" or "The" ("Fig. 3. Harness.", not "Fig. 3. The harness."). Applies to table titles too. | A, An, The | Applied automatically | ESM: Figure captions | `src/compose/ieee_text.py` | `caption_style` |
| FIG-12 | Figures, tables and equations mentioned INSIDE a caption follow the same wording rules as the text: "Fig. 2", "Table III", "(1)". |  | Applied automatically | ESM: Figure captions | `src/compose/ieee_text.py` | `caption_style` |

## Tables

| ID | Rule | Value | Status | Source | Applied in | Score check |
|---|---|---|---|---|---|---|
| TAB-01 | The table title goes ABOVE the table. |  | Applied automatically | GUIDE: Tables | `src/render/docx_renderer.py` | `caption_style` |
| TAB-02 | Tables are labelled "TABLE I" with a Roman numeral, centred. | TABLE | Applied automatically | ESM: Table captions | `src/render/docx_renderer.py` | `caption_style` |
| TAB-03 | The table title is in small capitals with no full stop at the end. |  | Applied automatically | ESM: Table captions | `src/render/docx_renderer.py` |  |
| TAB-04 | In the text a table is "Table I" (written in full, Roman numeral). |  | Applied automatically | GUIDE: Tables | `src/compose/ieee_text.py` | `visual_mentions` |
| TAB-05 | Tables use horizontal lines only. |  | Author's judgement | GUIDE: Tables |  |  |
| TAB-06 | Table footnotes use letters (a, b, c). |  | Author's judgement | GUIDE: Tables |  |  |

## Equations

| ID | Rule | Value | Status | Source | Applied in | Score check |
|---|---|---|---|---|---|---|
| EQN-01 | Equation numbers are in brackets, flush right: "(1)". |  | Applied automatically | ESM: Equations | `src/render/docx_renderer.py` |  |
| EQN-02 | In the text an equation is "(1)"; "Equation (1)" only at the start of a sentence. |  | Applied automatically | ESM: Equations | `src/compose/ieee_text.py` |  |
| EQN-03 | Equations are numbered consecutively, with no gaps or repeats. |  | Author's judgement | ESM: Equations |  |  |
| EQN-04 | Every symbol is defined; variables are in italics; an equation ending a sentence takes a full stop. |  | Author's judgement | GUIDE: Equations |  |  |

## Citations

| ID | Rule | Value | Status | Source | Applied in | Score check |
|---|---|---|---|---|---|---|
| CIT-01 | Citations are numbers in square brackets: [1]. |  | Applied automatically | REFGUIDE: Citing references | `src/compose/citations.py` | `citation_style` |
| CIT-02 | References are numbered in the order they are first cited. |  | Applied automatically | GUIDE: References | `src/compose/citations.py` | `reference_order` |
| CIT-03 | Several citations are written out one by one: "[3], [4], [5]". |  | Applied automatically | REFGUIDE: Citing references | `src/compose/citations.py` |  |
| CIT-04 | A citation sits inside the sentence punctuation: "...in [3]." |  | Applied automatically | REFGUIDE: Citing references | `src/compose/ieee_text.py` |  |
| CIT-05 | Write "in [3]", not "in reference [3]" or "Ref. [3]". |  | Applied automatically | GUIDE: References | `src/compose/ieee_text.py` |  |
| CIT-06 | Name three or more authors in the text as "Smith et al. [4]". |  | Author's judgement | REFGUIDE: Citing references |  |  |

## References

| ID | Rule | Value | Status | Source | Applied in | Score check |
|---|---|---|---|---|---|---|
| REF-01 | The list is numbered "[1]" flush left, with a hanging indent. |  | Applied automatically | REFGUIDE: Reference list | `src/render/docx_renderer.py` |  |
| REF-02 | Author initials come before the surname: "J. K. Author". |  | Applied automatically | REFGUIDE: Author names | `src/compose/references.py` | `reference_style` |
| REF-03 | More than six authors: the first author followed by "et al." | 6 | Applied automatically | REFGUIDE: Author names | `src/compose/references.py` |  |
| REF-04 | The article title is in double quotes, comma inside: "Title," |  | Applied automatically | REFGUIDE: Titles | `src/compose/references.py` | `reference_style` |
| REF-05 | Journal, conference and book names are in italics. |  | Applied automatically | REFGUIDE: Titles | `src/compose/references.py` |  |
| REF-06 | Journal and conference names are abbreviated with IEEE's word abbreviations ("Transactions" -> "Trans."). | 105 word abbreviations | Applied automatically | REFGUIDE: Abbreviations | `src/compose/references.py` | `venue_abbreviation` |
| REF-07 | Conference papers read "in Proc. ...". |  | Applied automatically | REFGUIDE: Conference papers | `src/compose/references.py` |  |
| REF-08 | Volume, issue and pages as "vol. 3, no. 2, pp. 10–20" with an en dash. |  | Applied automatically | REFGUIDE: Periodicals | `src/compose/references.py` |  |
| REF-09 | arXiv preprints: J. K. Author, "Title," year, arXiv:xxxx. |  | Applied automatically | REFGUIDE: Preprints | `src/compose/references.py` |  |
| REF-10 | Every reference has a year; "(n.d.)" when there is none. | (n.d.) | Applied automatically | REFGUIDE: Dates | `src/compose/references.py` |  |
| REF-11 | Every reference ends with a full stop, except one ending in a URL. |  | Applied automatically | REFGUIDE: Punctuation | `src/compose/references.py` |  |
| REF-12 | A DOI is written "doi: 10.xxxx."; an online source as "[Online]. Available: URL". |  | Applied automatically | REFGUIDE: Online sources | `src/compose/references.py` |  |

## Numbers and units

| ID | Rule | Value | Status | Source | Applied in | Score check |
|---|---|---|---|---|---|---|
| NUM-01 | A decimal has a zero before the point: "0.25", not ".25". |  | Applied automatically | ESM: Numbers | `src/compose/ieee_text.py` | `number_style` |
| NUM-02 | A space goes between a number and its unit: "10 mm". | 37 units | Applied automatically | ESM: Units | `src/compose/ieee_text.py` | `number_style` |
| NUM-03 | No space before the percent sign: "2%". |  | Applied automatically | ESM: Numbers | `src/compose/ieee_text.py` | `number_style` |
| NUM-04 | Ranges use an en dash; percentages repeat the sign: "40–50 mm", "20%–30%". |  | Applied automatically | ESM: Numbers | `src/compose/ieee_text.py` | `number_style` |
| NUM-05 | Numbers below 11 are spelled out in ordinary text. |  | Author's judgement | ESM: Numbers |  |  |
| NUM-06 | SI units are used; SI and CGS are not mixed. |  | Author's judgement | GUIDE: Units |  |  |
| NUM-07 | Dates are written day month year: "4 June 2002". |  | Author's judgement | ESM: Numbers |  |  |

## Writing style

| ID | Rule | Value | Status | Source | Applied in | Score check |
|---|---|---|---|---|---|---|
| STY-01 | No contractions: "do not", not "don't". | 27 contractions | Applied automatically | GUIDE: Other recommendations | `src/compose/ieee_text.py` |  |
| STY-02 | Every acronym is defined at its first use: "convolutional neural network (CNN)". |  | Checked and reported | ESM: Abbreviations and acronyms | `src/evaluate/compliance.py` | `acronyms` |
| STY-03 | One space after a full stop. |  | Applied automatically | GUIDE | `src/compose/ieee_text.py` |  |
| STY-04 | Use the serial comma: "A, B, and C". |  | Author's judgement | GUIDE |  |  |
| STY-05 | American spelling (Merriam-Webster). |  | Author's judgement | ESM |  |  |

## Back matter

| ID | Rule | Value | Status | Source | Applied in | Score check |
|---|---|---|---|---|---|---|
| ACK-01 | The heading is spelled "Acknowledgment" (no "e"). | Acknowledgment | Applied automatically | ESM: Acknowledgment | `src/structure/ieee_mapper.py` |  |
| ACK-02 | Order at the end: appendices, then Acknowledgment, then References. |  | Applied automatically | ESM: Acknowledgment | `src/render/docx_renderer.py` |  |
| ACK-03 | The acknowledgment is in the third person; funding goes in the first-page footnote. |  | Author's judgement | ESM: Acknowledgment |  |  |
