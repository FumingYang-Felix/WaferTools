#!/usr/bin/env python3
"""
WaferTools V3 Launcher with Animated Splash Screen
Copyright © 2025 Harvard University Lichtman Lab
"""

import sys
import os
import time
import threading
import subprocess
from pathlib import Path

# Check if tkinter is available for GUI splash screen
try:
    import tkinter as tk
    from tkinter import messagebox
    from PIL import Image, ImageTk, ImageEnhance
    TKINTER_AVAILABLE = True
except ImportError:
    TKINTER_AVAILABLE = False
    print("Warning: tkinter not available, using console mode")


class SplashScreen:
    """Animated splash screen with fade-in/fade-out effects"""
    
    def __init__(self, image_path, duration=3.0):
        self.image_path = image_path
        self.duration = duration
        self.root = None
        self.label = None
        self.original_image = None
        self.photo = None
        self.alpha = 0.0
        self.fade_speed = 0.05
        self.state = "fade_in"  # fade_in, display, fade_out, done
        self.start_time = time.time()
        
    def create_window(self):
        """Create the splash screen window"""
        self.root = tk.Tk()
        self.root.title("WaferTools V3")
        
        # Remove window decorations
        self.root.overrideredirect(True)
        
        # Load and prepare image
        try:
            self.original_image = Image.open(self.image_path)
            
            # Resize if too large
            max_size = (800, 800)
            self.original_image.thumbnail(max_size, Image.Resampling.LANCZOS)
            
            # Add padding and background
            width, height = self.original_image.size
            new_width = width + 100
            new_height = height + 150
            
            # Create white background
            background = Image.new('RGB', (new_width, new_height), 'white')
            
            # Paste logo centered
            x_offset = (new_width - width) // 2
            y_offset = 50
            
            if self.original_image.mode == 'RGBA':
                background.paste(self.original_image, (x_offset, y_offset), self.original_image)
            else:
                background.paste(self.original_image, (x_offset, y_offset))
            
            self.original_image = background
            
            # Create initial transparent version
            self.update_image()
            
            # Create label
            self.label = tk.Label(self.root, image=self.photo, bg='white')
            self.label.pack()
            
            # Add loading text
            loading_label = tk.Label(
                self.root, 
                text="Loading WaferTools V3...", 
                font=("Helvetica", 14, "bold"),
                bg='white',
                fg='#1e4d5f'
            )
            loading_label.pack(pady=10)
            
            # Progress bar simulation
            self.progress_label = tk.Label(
                self.root,
                text="● ○ ○",
                font=("Helvetica", 12),
                bg='white',
                fg='#1e4d5f'
            )
            self.progress_label.pack(pady=5)
            
            # Center window on screen
            self.root.update_idletasks()
            window_width = self.root.winfo_width()
            window_height = self.root.winfo_height()
            screen_width = self.root.winfo_screenwidth()
            screen_height = self.root.winfo_screenheight()
            
            x = (screen_width - window_width) // 2
            y = (screen_height - window_height) // 2
            
            self.root.geometry(f"+{x}+{y}")
            
            # Add shadow effect (rounded rectangle)
            self.root.attributes('-alpha', 0.0)  # Start fully transparent
            
        except Exception as e:
            print(f"Error loading splash image: {e}")
            self.root.destroy()
            raise
    
    def update_image(self):
        """Update image with current alpha value"""
        if self.original_image:
            # Create image with alpha
            enhanced = ImageEnhance.Brightness(self.original_image).enhance(self.alpha)
            self.photo = ImageTk.PhotoImage(enhanced)
            if self.label:
                self.label.configure(image=self.photo)
    
    def animate(self):
        """Animate the splash screen with fade effects"""
        elapsed = time.time() - self.start_time
        
        # Update progress indicator
        if hasattr(self, 'progress_label'):
            progress_states = ["● ○ ○", "● ● ○", "● ● ●", "○ ● ●", "○ ○ ●", "○ ○ ○"]
            idx = int((elapsed * 3) % len(progress_states))
            self.progress_label.configure(text=progress_states[idx])
        
        if self.state == "fade_in":
            # Fade in
            self.alpha = min(1.0, self.alpha + self.fade_speed)
            self.root.attributes('-alpha', self.alpha)
            
            if self.alpha >= 1.0:
                self.state = "display"
                self.display_start = time.time()
        
        elif self.state == "display":
            # Display for duration
            if time.time() - self.display_start >= self.duration - 1.0:
                self.state = "fade_out"
        
        elif self.state == "fade_out":
            # Fade out
            self.alpha = max(0.0, self.alpha - self.fade_speed)
            self.root.attributes('-alpha', self.alpha)
            
            if self.alpha <= 0.0:
                self.state = "done"
                self.root.destroy()
                return
        
        # Schedule next frame
        self.root.after(50, self.animate)
    
    def show(self):
        """Show the splash screen"""
        try:
            self.create_window()
            self.animate()
            self.root.mainloop()
        except Exception as e:
            print(f"Splash screen error: {e}")


def console_splash(duration=3.0):
    """Console-based splash screen for systems without GUI"""
    print("\n" + "="*60)
    print("""
    ██╗    ██╗ █████╗ ███████╗███████╗██████╗ ████████╗ ██████╗  ██████╗ ██╗     ███████╗
    ██║    ██║██╔══██╗██╔════╝██╔════╝██╔══██╗╚══██╔══╝██╔═══██╗██╔═══██╗██║     ██╔════╝
    ██║ █╗ ██║███████║█████╗  █████╗  ██████╔╝   ██║   ██║   ██║██║   ██║██║     ███████╗
    ██║███╗██║██╔══██║██╔══╝  ██╔══╝  ██╔══██╗   ██║   ██║   ██║██║   ██║██║     ╚════██║
    ╚███╔███╔╝██║  ██║██║     ███████╗██║  ██║   ██║   ╚██████╔╝╚██████╔╝███████╗███████║
     ╚══╝╚══╝ ╚═╝  ╚═╝╚═╝     ╚══════╝╚═╝  ╚═╝   ╚═╝    ╚═════╝  ╚═════╝ ╚══════╝╚══════╝
    """)
    print("                           Version 3.0")
    print("              (c) 2025 Harvard University Lichtman Lab")
    print("="*60)
    print("\n[*] Loading WaferTools...\n")
    
    # Animated loading bar
    bar_length = 40
    for i in range(bar_length + 1):
        percent = (i / bar_length) * 100
        filled = '█' * i
        empty = '░' * (bar_length - i)
        print(f"\r[{filled}{empty}] {percent:.0f}%", end='', flush=True)
        time.sleep(duration / bar_length)
    
    print("\n\n[*] Ready!\n")


def check_dependencies():
    """Check if required dependencies are installed"""
    print("[*] Checking dependencies...")
    
    missing = []
    required = [
        'dash', 'pandas', 'numpy', 'cv2', 'torch', 
        'segment_anything', 'PIL', 'plotly'
    ]
    
    for package in required:
        try:
            if package == 'cv2':
                __import__('cv2')
            elif package == 'PIL':
                __import__('PIL')
            else:
                __import__(package)
            print(f"  [+] {package}")
        except ImportError:
            missing.append(package)
            print(f"  [-] {package} - MISSING")
    
    if missing:
        print(f"\n[!] Missing dependencies: {', '.join(missing)}")
        print("    Please install: pip install -r requirements.txt")
        return False
    
    print("[+] All dependencies installed\n")
    return True


def check_sam_weights():
    """Check if SAM model weights are available"""
    print("[*] Checking SAM model weights...")
    
    base_dir = Path(__file__).parent
    models = ['sam_vit_h.pth', 'sam_vit_l.pth', 'sam_vit_b.pth']
    
    found = []
    for model in models:
        if (base_dir / model).exists():
            found.append(model)
            print(f"  [+] {model}")
    
    if not found:
        print("  [!] No SAM weights found")
        print("      They will be downloaded on first use")
    else:
        print(f"[+] Found {len(found)} SAM model(s)\n")
    
    return True


def start_wafertools():
    """Start the WaferTools application"""
    print("[*] Starting WaferTools V3...\n")
    
    # Get the directory of this script
    base_dir = Path(__file__).parent
    app_path = base_dir / "app.py"
    
    if not app_path.exists():
        print(f"[!] Error: app.py not found at {app_path}")
        return False
    
    try:
        # Launch the Dash app
        print("[*] Launching Dash application...")
        print("    URL: http://127.0.0.1:8050")
        print("\n[*] Opening browser...")
        print("    Press Ctrl+C to stop the server\n")
        print("-" * 60 + "\n")
        
        # Open browser after a short delay
        def open_browser():
            time.sleep(2)  # Wait for server to start
            import webbrowser
            webbrowser.open('http://127.0.0.1:8050')
        
        browser_thread = threading.Thread(target=open_browser, daemon=True)
        browser_thread.start()
        
        # Run the app
        subprocess.run([sys.executable, str(app_path)], check=True)
        
    except KeyboardInterrupt:
        print("\n\n[*] Shutting down WaferTools...")
        return True
    except Exception as e:
        print(f"\n[!] Error starting application: {e}")
        return False


def main():
    """Main launcher function"""
    # Print header
    print("\n" + "="*60)
    print("  WaferTools V3 Launcher")
    print("  Harvard University Lichtman Lab")
    print("="*60 + "\n")
    
    # Change to script directory
    os.chdir(Path(__file__).parent)
    
    # Show splash screen
    image_path = Path(__file__).parent / "WT_software_page1.png"
    
    # Use console splash for better compatibility
    # (GUI splash has threading issues on some macOS versions)
    if image_path.exists():
        print("[*] Loading WaferTools...\n")
        console_splash(duration=2.0)
    else:
        print(f"[!] Logo not found: {image_path}\n")
        console_splash(duration=2.0)
    
    # Check dependencies
    deps_ok = check_dependencies()
    
    if not deps_ok:
        print("\n[!] Dependency check failed.")
        print("    Please run: pip install -r requirements.txt")
        print("\n[*] Press Ctrl+C to exit or close this window")
        # Keep the window open
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            return 1
    
    # Check SAM weights
    check_sam_weights()
    
    # Start the application
    success = start_wafertools()
    
    if not success:
        print("\n[!] Failed to start WaferTools")
        print("[*] Press Ctrl+C to exit or close this window")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            return 1
    
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n[*] Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"\n[!] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        print("\n[*] Press Ctrl+C to exit or close this window")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            sys.exit(1)
