with open("frontend/src/components/ReferenceVerification.js", "r", encoding="utf-8") as f:
    content = f.read()

import re

# Replace the extraction
content = re.sub(
    r"// Extract a clean title from the match object.*?const matchedAuthor =.*?: null;",
    """  const matchedTitle = v.matched_title;
  const matchedDOI = v.matched_doi;
  const matchedAuthor = v.matched_authors;
  const matchedArxiv = v.matched_arxiv;
  const matchedYear = v.matched_year;
  const matchedSource = v.matched_source || 'Crossref';""",
    content,
    flags=re.DOTALL
)

# Replace the UI rendering
new_ui = """                {matchedTitle && (
                  <div className="detail-section">
                    <h4 className="detail-heading">📖 Matched Record ({matchedSource})</h4>
                    <div className="detail-row detail-reason">
                      <span className="detail-label">Title</span>
                      <span className="detail-value-text">{matchedTitle}</span>
                    </div>
                    {matchedAuthor && (
                      <div className="detail-row">
                        <span className="detail-label">Author</span>
                        <span className="detail-value-text">{matchedAuthor}</span>
                      </div>
                    )}
                    {matchedYear && (
                      <div className="detail-row">
                        <span className="detail-label">Year</span>
                        <span className="detail-value-text">{matchedYear}</span>
                      </div>
                    )}
                    {matchedSource && (
                      <div className="detail-row">
                        <span className="detail-label">Source</span>
                        <span className="detail-value-text">{matchedSource}</span>
                      </div>
                    )}
                    {matchedArxiv && (
                      <div className="detail-row">
                        <span className="detail-label">arXiv ID</span>
                        <a
                          className="detail-value-text"
                          href={`https://arxiv.org/abs/${matchedArxiv}`}
                          target="_blank"
                          rel="noopener noreferrer"
                          onClick={(e) => e.stopPropagation()}
                          style={{ color: '#1e40af', textDecoration: 'underline' }}
                        >
                          {matchedArxiv}
                        </a>
                      </div>
                    )}
                    {matchedDOI && (
                      <div className="detail-row">
                        <span className="detail-label">DOI</span>
                        <a
                          className="detail-value-text"
                          href={
                            matchedDOI.startsWith('http')
                              ? matchedDOI
                              : `https://doi.org/${matchedDOI}`
                          }
                          target="_blank"
                          rel="noopener noreferrer"
                          onClick={(e) => e.stopPropagation()}
                          style={{ color: '#1e40af', textDecoration: 'underline' }}
                        >
                          {matchedDOI}
                        </a>
                      </div>
                    )}
                  </div>
                )}"""

# It's tricky to regex replace JSX exactly because of emojis and escaping.
content = re.sub(
    r"\{matchedTitle && \([\s\S]*?className=\"detail-heading\">.*?Matched Record \(Crossref\)</h4>[\s\S]*?</div>\s*\)\}",
    new_ui,
    content
)


with open("frontend/src/components/ReferenceVerification.js", "w", encoding="utf-8") as f:
    f.write(content)
