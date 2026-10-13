const fs = require('fs');
const babel = require('@babel/parser');
const code = fs.readFileSync('frontend/src/components/ReferenceVerification.js', 'utf8');
try {
  babel.parse(code, {
    sourceType: 'module',
    plugins: ['jsx']
  });
  console.log("No syntax errors!");
} catch (e) {
  console.log(e);
}
