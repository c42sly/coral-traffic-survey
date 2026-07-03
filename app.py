import controller
import web_gui

if __name__ == '__main__':
    print("Starting threads...")
    controller.start()
    
    print("Starting web server...")
    try:
        web_gui.app.run(host='0.0.0.0', port=5000, threaded=True)
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
    finally:
        controller.stop()
