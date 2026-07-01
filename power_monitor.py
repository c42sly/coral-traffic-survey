import subprocess
import threading
import shared

# Make sure to add this to shared.py:
# current_power_watts = 0.0

def power_worker():
    print("🔋 Starting Power Monitor Thread...")
    
    # We only request Wattage to make parsing easy. 
    # Ensure the 'um25c' binary is executable and in the right path.
    cmd = ["sudo", "./um25c", "-d", "/dev/rfcomm0", "-f", "Watt"]
    
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        
        while True:
            line = proc.stdout.readline()
            if not line:
                break
                
            try:
                # The binary outputs the requested format followed by a newline
                watts = float(line.strip())
                with shared.lock:
                    shared.current_power_watts = watts
            except ValueError:
                pass # Ignore malformed lines while Bluetooth syncs
                
    except Exception as e:
        print(f"Power monitor failed to start: {e}")
