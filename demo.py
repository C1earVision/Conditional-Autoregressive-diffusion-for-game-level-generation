"""
Interactive Desktop Demo App for Conditional Autoregressive Diffusion Level Generation

This Tkinter-based desktop app allows users to:
- Adjust generation parameters (difficulty, temperature, guidance scale, patches)
- Generate Super Mario Bros levels using the existing generation script
- Visualize generated levels in the application window
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import yaml
import subprocess
import os
import glob
import threading
from PIL import Image, ImageTk


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(PROJECT_DIR, "config", "generation_config.yaml")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "output", "generated_levels")


class LevelGeneratorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🍄 Mario Level Generator")
        self.root.geometry("1000x700")
        self.root.minsize(800, 600)
        

        self.style = ttk.Style()
        self.style.theme_use('clam')
        

        self.root.configure(bg='#2b2b2b')
        self.style.configure('TFrame', background='#2b2b2b')
        self.style.configure('TLabel', background='#2b2b2b', foreground='#ffffff', font=('Segoe UI', 10))
        self.style.configure('Title.TLabel', font=('Segoe UI', 16, 'bold'), foreground='#ff9500')
        self.style.configure('Header.TLabel', font=('Segoe UI', 12, 'bold'), foreground='#ffffff')
        self.style.configure('TButton', font=('Segoe UI', 11, 'bold'))
        self.style.configure('Generate.TButton', font=('Segoe UI', 12, 'bold'))
        self.style.configure('TScale', background='#2b2b2b')
        

        self.difficulty_var = tk.DoubleVar(value=1.0)
        self.temperature_var = tk.DoubleVar(value=0.2)
        self.guidance_var = tk.DoubleVar(value=3.0)
        self.patches_var = tk.IntVar(value=1)
        

        self.status_var = tk.StringVar(value="Ready to generate")
        self.full_error = "" 
        

        self.current_image = None
        

        self.create_widgets()
        

        os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    def create_widgets(self):

        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        

        title_label = ttk.Label(
            main_frame, 
            text="🍄 Conditional Autoregressive Diffusion Level Generator",
            style='Title.TLabel'
        )
        title_label.pack(pady=(0, 15))
        

        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True)
        

        left_panel = ttk.Frame(content_frame, padding="10")
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        

        ttk.Label(left_panel, text="⚙️ Generation Parameters", style='Header.TLabel').pack(anchor=tk.W, pady=(0, 15))
        

        self.create_slider(
            left_panel,
            "🎯 Difficulty Target",
            "0.0 = Easy, 1.0 = Hard",
            self.difficulty_var,
            0.0, 1.0, 0.05
        )
        

        self.create_slider(
            left_panel,
            "🌡️ Temperature",
            "Higher = more random",
            self.temperature_var,
            0.1, 2.0, 0.1
        )
        

        self.create_slider(
            left_panel,
            "🧭 Guidance Scale (CFG)",
            "Higher = stronger conditioning",
            self.guidance_var,
            1.0, 10.0, 0.5
        )
        

        self.create_slider(
            left_panel,
            "📏 Number of Patches",
            "Controls level length",
            self.patches_var,
            1, 30, 1,
            is_int=True
        )
        

        self.generate_btn = ttk.Button(
            left_panel,
            text="🎮 Generate Level",
            style='Generate.TButton',
            command=self.start_generation
        )
        self.generate_btn.pack(fill=tk.X, pady=(20, 10))
        

        status_frame = ttk.Frame(left_panel)
        status_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        
        ttk.Label(status_frame, text="Status:", style='Header.TLabel').pack(anchor=tk.W, pady=(0, 5))
        

        self.status_label = ttk.Label(
            status_frame, 
            textvariable=self.status_var,
            wraplength=250
        )
        self.status_label.pack(anchor=tk.W)
        

        self.details_btn = ttk.Button(
            status_frame,
            text="📋 View Full Error",
            command=self.show_error_details
        )
        

        right_panel = ttk.Frame(content_frame, padding="10")
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        ttk.Label(right_panel, text="🎮 Generated Level", style='Header.TLabel').pack(anchor=tk.W, pady=(0, 10))
        

        image_frame = tk.Frame(right_panel, bg='#404040', bd=2, relief=tk.SUNKEN)
        image_frame.pack(fill=tk.BOTH, expand=True)
        
        self.image_label = tk.Label(
            image_frame, 
            text="Generated level will appear here",
            bg='#1a1a1a',
            fg='#888888',
            font=('Segoe UI', 12)
        )
        self.image_label.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
    
    def create_slider(self, parent, label, info, variable, min_val, max_val, step, is_int=False):
        """Create a labeled slider with value display."""
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, pady=(0, 15))
        

        ttk.Label(frame, text=label, style='Header.TLabel').pack(anchor=tk.W)
        ttk.Label(frame, text=info, font=('Segoe UI', 9), foreground='#888888').pack(anchor=tk.W)
        

        slider_frame = ttk.Frame(frame)
        slider_frame.pack(fill=tk.X, pady=(5, 0))
        

        if is_int:
            value_label = ttk.Label(slider_frame, text=f"{int(variable.get())}", width=5)
        else:
            value_label = ttk.Label(slider_frame, text=f"{variable.get():.2f}", width=5)
        value_label.pack(side=tk.RIGHT)
        

        def update_label(*args):
            if is_int:
                value_label.config(text=f"{int(variable.get())}")
            else:
                value_label.config(text=f"{variable.get():.2f}")
        
        variable.trace_add('write', update_label)
        
        slider = ttk.Scale(
            slider_frame,
            from_=min_val,
            to=max_val,
            variable=variable,
            orient=tk.HORIZONTAL
        )
        slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
    
    def show_error_details(self):
        """Show full error message in a popup window."""
        if not self.full_error:
            return
        

        popup = tk.Toplevel(self.root)
        popup.title("Error Details")
        popup.geometry("700x400")
        popup.configure(bg='#2b2b2b')
        

        text_frame = ttk.Frame(popup, padding="10")
        text_frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(text_frame, text="Full Error Output:", style='Header.TLabel').pack(anchor=tk.W, pady=(0, 10))
        
        text_widget = scrolledtext.ScrolledText(
            text_frame,
            wrap=tk.WORD,
            font=('Consolas', 9),
            bg='#1a1a1a',
            fg='#ff6b6b',
            insertbackground='#ffffff'
        )
        text_widget.pack(fill=tk.BOTH, expand=True)
        text_widget.insert('1.0', self.full_error)
        text_widget.config(state=tk.DISABLED)
        

        close_btn = ttk.Button(popup, text="Close", command=popup.destroy)
        close_btn.pack(pady=10)
    
    def update_config(self):
        """Update generation_config.yaml with current slider values."""
        with open(CONFIG_PATH, 'r') as f:
            config = yaml.safe_load(f)
        
        config['generation']['difficulty_target'] = float(self.difficulty_var.get())
        config['generation']['temperature'] = float(self.temperature_var.get())
        config['generation']['guidance_scale'] = float(self.guidance_var.get())
        config['generation']['patches_per_level'] = int(self.patches_var.get())
        config['generation']['num_levels'] = 1
        
        with open(CONFIG_PATH, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    
    def get_latest_level_image(self):
        """Get the most recently generated level image (PNG visualization only)."""
        pattern = os.path.join(OUTPUT_DIR, "level_*_visual.png")
        files = glob.glob(pattern)
        if not files:

            pattern = os.path.join(OUTPUT_DIR, "level_*.png")
            all_files = glob.glob(pattern)
            files = [f for f in all_files if '_text' not in f.lower()]
        
        if files:
            latest = max(files, key=os.path.getmtime)
            return latest
        return None
    
    def start_generation(self):
        """Start level generation in a separate thread."""
        self.generate_btn.config(state=tk.DISABLED)
        self.status_var.set("⏳ Generating level...")
        self.details_btn.pack_forget()  # Hide details button
        self.full_error = ""  # Clear previous error
        

        thread = threading.Thread(target=self.generate_level)
        thread.daemon = True
        thread.start()
    
    def generate_level(self):
        """Generate a level using the existing script."""
        try:
            self.update_config()
            
            self.root.after(0, lambda: self.status_var.set("⏳ Running generation script..."))
            

            result = subprocess.run(
                ["python", "-m", "scripts.generate_levels"],
                cwd=PROJECT_DIR,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.returncode != 0:

                self.full_error = f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
                

                error_preview = result.stderr[:100] if result.stderr else "Unknown error"
                error_msg = f"✗ Generation failed: {error_preview}..."
                
                self.root.after(0, lambda: self.status_var.set(error_msg))
                self.root.after(0, lambda: self.details_btn.pack(anchor=tk.W, pady=(5, 0)))
                self.root.after(0, lambda: self.generate_btn.config(state=tk.NORMAL))
                return
            

            image_path = self.get_latest_level_image()
            
            if image_path:
                self.root.after(0, lambda: self.display_image(image_path))
                status = f"✓ Generated! Difficulty: {self.difficulty_var.get():.2f} | Temp: {self.temperature_var.get():.2f}"
                self.root.after(0, lambda: self.status_var.set(status))
            else:
                self.root.after(0, lambda: self.status_var.set("✗ No image found in output folder"))
            
        except subprocess.TimeoutExpired:
            self.full_error = "Generation process exceeded 300 second timeout"
            self.root.after(0, lambda: self.status_var.set("✗ Generation timed out (300s limit)"))
            self.root.after(0, lambda: self.details_btn.pack(anchor=tk.W, pady=(5, 0)))
        except Exception as e:
            self.full_error = f"Exception: {type(e).__name__}\n\n{str(e)}"
            error_preview = str(e)[:80]
            self.root.after(0, lambda: self.status_var.set(f"✗ Error: {error_preview}..."))
            self.root.after(0, lambda: self.details_btn.pack(anchor=tk.W, pady=(5, 0)))
        finally:
            self.root.after(0, lambda: self.generate_btn.config(state=tk.NORMAL))
    
    def display_image(self, image_path):
        """Display the generated level image."""
        try:
            img = Image.open(image_path)
            

            label_width = self.image_label.winfo_width()
            label_height = self.image_label.winfo_height()
            
            if label_width > 1 and label_height > 1:

                img_ratio = img.width / img.height
                label_ratio = label_width / label_height
                
                if img_ratio > label_ratio:
                    new_width = label_width - 20
                    new_height = int(new_width / img_ratio)
                else:
                    new_height = label_height - 20
                    new_width = int(new_height * img_ratio)
                
                img = img.resize((max(1, new_width), max(1, new_height)), Image.Resampling.LANCZOS)
            
            self.current_image = ImageTk.PhotoImage(img)
            self.image_label.config(image=self.current_image, text="")
            
        except Exception as e:
            self.status_var.set(f"✗ Error loading image: {str(e)}")


def main():
    root = tk.Tk()
    app = LevelGeneratorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()