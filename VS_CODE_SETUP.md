# How to Run This Project in VS Code

## Prerequisites
1. Make sure you have VS Code installed
2. Install the **Python extension** for VS Code (if not already installed)
   - Open VS Code
   - Go to Extensions (Ctrl+Shift+X)
   - Search for "Python" by Microsoft
   - Click Install

## Step-by-Step Instructions

### Method 1: Using VS Code Terminal (Recommended)

1. **Open the project folder in VS Code**
   - File → Open Folder
   - Navigate to: `C:\Users\harsh\OneDrive\Desktop\AI ML\Langchain models\chatbot`
   - Click Select Folder

2. **Select the Python interpreter**
   - Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on Mac)
   - Type: "Python: Select Interpreter"
   - Choose the interpreter from: `.\env\Scripts\python.exe`
   - Or manually browse and select: `env\Scripts\python.exe`

3. **Open the integrated terminal**
   - Press `` Ctrl+` `` (backtick) or go to Terminal → New Terminal
   - The terminal should automatically activate the virtual environment

4. **Run the Streamlit app**
   - In the terminal, type:
     ```powershell
     streamlit run app.py
     ```
   - OR use:
     ```powershell
     python -m streamlit run app.py
     ```

5. **Access the app**
   - The terminal will show a local URL (usually `http://localhost:8501`)
   - Click the link or copy it to your browser
   - The app should open automatically in your default browser

### Method 2: Using VS Code Debug/Run Configuration

1. **Open the project in VS Code** (as in Method 1, Step 1)

2. **Select the Python interpreter** (as in Method 1, Step 2)

3. **Run using Debug**
   - Go to Run and Debug panel (Ctrl+Shift+D)
   - Select "Python: Streamlit" from the dropdown
   - Click the green play button (or press F5)
   - The app will start and open in your browser

### Method 3: Create a Task (Alternative)

1. **Open Command Palette**: `Ctrl+Shift+P`

2. **Type**: "Tasks: Configure Task"

3. **Create a new task** with:
   ```json
   {
       "version": "2.0.0",
       "tasks": [
           {
               "label": "Run Streamlit",
               "type": "shell",
               "command": "${workspaceFolder}/env/Scripts/python.exe -m streamlit run app.py",
               "group": {
                   "kind": "build",
                   "isDefault": true
               },
               "presentation": {
                   "echo": true,
                   "reveal": "always",
                   "focus": false,
                   "panel": "new"
               }
           }
       ]
   }
   ```

4. **Run the task**: `Ctrl+Shift+P` → "Tasks: Run Task" → "Run Streamlit"

## Troubleshooting

### Issue: "streamlit is not recognized"
**Solution**: Make sure you've selected the correct Python interpreter from the `env` folder.

### Issue: Module not found errors
**Solution**: 
1. Verify the interpreter is set to `env\Scripts\python.exe`
2. Open terminal and run:
   ```powershell
   .\env\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

### Issue: Terminal doesn't activate environment automatically
**Solution**: 
1. Manually activate in terminal:
   ```powershell
   .\env\Scripts\Activate.ps1
   ```
2. Or use the full path:
   ```powershell
   .\env\Scripts\python.exe -m streamlit run app.py
   ```

### Issue: PowerShell execution policy error
**Solution**: If you see an execution policy error, run:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

## Quick Start Command

Once VS Code is set up, you can simply run in the terminal:
```powershell
.\env\Scripts\python.exe -m streamlit run app.py
```

## Note

The `.vscode/settings.json` file has been created to automatically:
- Set the Python interpreter to your virtual environment
- Auto-activate the environment in terminals
- Configure proper Python paths

The `.vscode/launch.json` file allows you to run/debug the Streamlit app directly from VS Code's Run panel.


