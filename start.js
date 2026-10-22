const { spawn } = require('child_process');
const fs = require('fs');
const path = require('path');

// Determine Python executable inside venv
const rootDir = __dirname;
let pythonPath = 'python';

const winVenv = path.join(rootDir, 'venv', 'Scripts', 'python.exe');
const unixVenv = path.join(rootDir, 'venv', 'bin', 'python');

if (process.platform === 'win32' && fs.existsSync(winVenv)) {
  pythonPath = winVenv;
} else if (fs.existsSync(unixVenv)) {
  pythonPath = unixVenv;
}

console.log('=====================================================');
console.log(' 🛡️ Starting PhishGuard Streamlit Server...');
console.log(` Using Python: ${pythonPath}`);
console.log(' Local URL:    http://localhost:8501');
console.log(' Press Ctrl+C in terminal to stop the server');
console.log('=====================================================\n');

const child = spawn(
  pythonPath,
  ['-m', 'streamlit', 'run', 'app/app.py', '--server.headless=true', '--server.port=8501', ...process.argv.slice(2)],
  {
    stdio: 'inherit',
    shell: true,
    env: {
      ...process.env,
      PYTHONIOENCODING: 'utf-8',
      PYTHONUTF8: '1'
    }
  }
);

child.on('exit', (code) => {
  process.exit(code || 0);
});
