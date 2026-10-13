with open("frontend/src/components/ReferenceVerification.js", "r", encoding="utf-8") as f:
    content = f.read()

duplicate_block = """                    {matchedDOI && (
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

parts = content.split(duplicate_block)
if len(parts) == 3:
    # Remove the second occurrence
    content = parts[0] + duplicate_block + parts[1] + parts[2]
elif len(parts) > 3:
    print(f"Unexpected number of occurrences: {len(parts)-1}")

with open("frontend/src/components/ReferenceVerification.js", "w", encoding="utf-8") as f:
    f.write(content)
