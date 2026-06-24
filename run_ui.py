import sys
import os

if __name__ == "__main__":
    # Гарантируем, что корневая папка проекта находится в PYTHONPATH
    project_root = os.path.dirname(os.path.abspath(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
        
    try:
        from streamlit.web import cli as stcli
    except ImportError:
        print("[ОШИБКА] Streamlit не установлен. Пожалуйста, запустите setup.bat или установите streamlit через pip.")
        sys.exit(1)
        
    sys.argv = ["streamlit", "run", "src/ui/app.py"]
    sys.exit(stcli.main())
