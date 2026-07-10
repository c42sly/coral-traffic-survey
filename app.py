import controller
import web_gui
import traceback

if __name__ == '__main__':
    print("Starting threads...")
    try:
        controller.start()
        print("Starting web server...")
        web_gui.app.run(host='0.0.0.0', port=5000, threaded=True)
    except Exception as e:
        print(f"FATAL ERROR: {e}")
        traceback.print_exc()
    finally:
        print("Main thread exiting — calling controller.stop()")
        controller.stop()
