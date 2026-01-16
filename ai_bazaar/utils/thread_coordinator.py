"""
Simple thread coordinator with GUI for controlling Thread A and Thread B.
"""

import tkinter as tk
from tkinter import ttk, scrolledtext
import threading
import time
from typing import Optional
from .thread_manager import ThreadManager, ThreadState


class ThreadCoordinator:
    """
    Simple GUI-based thread coordinator for managing Thread A and Thread B.
    """
    
    def __init__(self, thread_manager: ThreadManager):
        self.thread_manager = thread_manager
        self.root = None
        self.is_running = False
        
        # GUI elements
        self.status_label_a = None
        self.status_label_b = None
        self.start_button_a = None
        self.pause_button_a = None
        self.stop_button_a = None
        self.start_button_b = None
        self.pause_button_b = None
        self.stop_button_b = None
        self.log_text = None
        
    def show_gui(self):
        """Show the thread coordinator GUI."""
        self.is_running = True
        self.root = tk.Tk()
        self.root.title("Thread Coordinator")
        self.root.geometry("600x500")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Create main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(3, weight=1)
        
        # Thread A controls
        ttk.Label(main_frame, text="Thread A (Simulation):", font=("Arial", 12, "bold")).grid(row=0, column=0, columnspan=3, sticky=tk.W, pady=(0, 10))
        
        self.status_label_a = ttk.Label(main_frame, text="Status: STOPPED", foreground="red")
        self.status_label_a.grid(row=1, column=0, sticky=tk.W, pady=(0, 5))
        
        button_frame_a = ttk.Frame(main_frame)
        button_frame_a.grid(row=1, column=1, columnspan=2, sticky=tk.E, pady=(0, 5))
        
        self.start_button_a = ttk.Button(button_frame_a, text="Start", command=self.start_thread_a)
        self.start_button_a.grid(row=0, column=0, padx=(0, 5))
        
        self.pause_button_a = ttk.Button(button_frame_a, text="Pause", command=self.pause_thread_a, state=tk.DISABLED)
        self.pause_button_a.grid(row=0, column=1, padx=(0, 5))
        
        self.stop_button_a = ttk.Button(button_frame_a, text="Stop", command=self.stop_thread_a, state=tk.DISABLED)
        self.stop_button_a.grid(row=0, column=2)
        
        # Thread B controls
        ttk.Label(main_frame, text="Thread B (Conversations):", font=("Arial", 12, "bold")).grid(row=2, column=0, columnspan=3, sticky=tk.W, pady=(10, 10))
        
        self.status_label_b = ttk.Label(main_frame, text="Status: STOPPED", foreground="red")
        self.status_label_b.grid(row=3, column=0, sticky=tk.W, pady=(0, 5))
        
        button_frame_b = ttk.Frame(main_frame)
        button_frame_b.grid(row=3, column=1, columnspan=2, sticky=tk.E, pady=(0, 5))
        
        self.start_button_b = ttk.Button(button_frame_b, text="Start", command=self.start_thread_b, state=tk.DISABLED)
        self.start_button_b.grid(row=0, column=0, padx=(0, 5))
        
        self.pause_button_b = ttk.Button(button_frame_b, text="Pause", command=self.pause_thread_b, state=tk.DISABLED)
        self.pause_button_b.grid(row=0, column=1, padx=(0, 5))
        
        self.stop_button_b = ttk.Button(button_frame_b, text="Stop", command=self.stop_thread_b, state=tk.DISABLED)
        self.stop_button_b.grid(row=0, column=2)
        
        # Log area
        ttk.Label(main_frame, text="Log:", font=("Arial", 10, "bold")).grid(row=4, column=0, columnspan=3, sticky=tk.W, pady=(10, 5))
        
        self.log_text = scrolledtext.ScrolledText(main_frame, height=15, wrap=tk.WORD)
        self.log_text.grid(row=5, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        
        # Start status update loop
        self.update_status()
        
        # Center the window
        self.root.update_idletasks()
        x = (self.root.winfo_screenwidth() // 2) - (self.root.winfo_width() // 2)
        y = (self.root.winfo_screenheight() // 2) - (self.root.winfo_height() // 2)
        self.root.geometry(f"+{x}+{y}")
        
        # Start the GUI main loop in the main thread
        # Note: This will block until the GUI is closed
        self.run_gui()
        
    def run_gui(self):
        """Run the GUI main loop."""
        try:
            self.root.mainloop()
        except Exception as e:
            # Don't try to log from this thread - it causes tkinter issues
            print(f"GUI error: {e}")
        finally:
            self.is_running = False
    
            
    def start_thread_a(self):
        """Start Thread A."""
        try:
            self.thread_manager.start_thread_a()
            self.log("Thread A started")
        except Exception as e:
            self.log(f"Error starting Thread A: {e}")
            
    def pause_thread_a(self):
        """Pause Thread A."""
        try:
            self.thread_manager.pause_thread_a()
            self.log("Thread A paused")
        except Exception as e:
            self.log(f"Error pausing Thread A: {e}")
            
    def stop_thread_a(self):
        """Stop Thread A."""
        try:
            self.thread_manager.stop_thread_a()
            self.log("Thread A stopped")
        except Exception as e:
            self.log(f"Error stopping Thread A: {e}")
            
    def start_thread_b(self):
        """Start Thread B."""
        try:
            self.thread_manager.start_thread_b()
            self.log("Thread B started")
        except Exception as e:
            self.log(f"Error starting Thread B: {e}")
            
    def pause_thread_b(self):
        """Pause Thread B."""
        try:
            self.thread_manager.pause_thread_b()
            self.log("Thread B paused")
        except Exception as e:
            self.log(f"Error pausing Thread B: {e}")
            
    def stop_thread_b(self):
        """Stop Thread B."""
        try:
            self.thread_manager.stop_thread_b()
            self.log("Thread B stopped")
        except Exception as e:
            self.log(f"Error stopping Thread B: {e}")
            
    def update_status(self):
        """Update the status labels and button states."""
        if not self.is_running or not self.root:
            return
            
        try:
            # Update Thread A status
            state_a = self.thread_manager.thread_a_state
            if state_a == ThreadState.RUNNING:
                self.status_label_a.config(text="Status: RUNNING", foreground="green")
                self.start_button_a.config(state=tk.DISABLED)
                self.pause_button_a.config(state=tk.NORMAL)
                self.stop_button_a.config(state=tk.NORMAL)
            elif state_a == ThreadState.PAUSED:
                self.status_label_a.config(text="Status: PAUSED", foreground="orange")
                self.start_button_a.config(state=tk.NORMAL)
                self.pause_button_a.config(state=tk.DISABLED)
                self.stop_button_a.config(state=tk.NORMAL)
            else:  # STOPPED
                self.status_label_a.config(text="Status: STOPPED", foreground="red")
                self.start_button_a.config(state=tk.NORMAL)
                self.pause_button_a.config(state=tk.DISABLED)
                self.stop_button_a.config(state=tk.DISABLED)
                
            # Update Thread B status
            state_b = self.thread_manager.thread_b_state
            if state_b == ThreadState.RUNNING:
                self.status_label_b.config(text="Status: RUNNING", foreground="green")
                self.start_button_b.config(state=tk.DISABLED)
                self.pause_button_b.config(state=tk.NORMAL)
                self.stop_button_b.config(state=tk.NORMAL)
            elif state_b == ThreadState.PAUSED:
                self.status_label_b.config(text="Status: PAUSED", foreground="orange")
                self.start_button_b.config(state=tk.NORMAL)
                self.pause_button_b.config(state=tk.DISABLED)
                self.stop_button_b.config(state=tk.NORMAL)
            else:  # STOPPED
                self.status_label_b.config(text="Status: STOPPED", foreground="red")
                self.start_button_b.config(state=tk.NORMAL)
                self.pause_button_b.config(state=tk.DISABLED)
                self.stop_button_b.config(state=tk.DISABLED)
                
            # Enable/disable Thread B start button based on Thread A state
            if state_a == ThreadState.PAUSED and state_b == ThreadState.STOPPED:
                self.start_button_b.config(state=tk.NORMAL)
            elif state_a != ThreadState.PAUSED:
                self.start_button_b.config(state=tk.DISABLED)
                
        except Exception as e:
            self.log(f"Error updating status: {e}")
            
        # Schedule next update
        if self.is_running and self.root:
            self.root.after(500, self.update_status)  # Update every 500ms
            
    
    def log(self, message: str):
        """Add a message to the log."""
        if self.log_text:
            timestamp = time.strftime("%H:%M:%S")
            self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
            self.log_text.see(tk.END)
            
    def on_closing(self):
        """Handle window closing."""
        self.is_running = False
        if self.root:
            try:
                self.root.destroy()
            except tk.TclError:
                # GUI already destroyed, ignore the error
                pass
            
    def close_gui(self):
        """Close the GUI programmatically."""
        self.is_running = False
        if self.root:
            try:
                self.root.quit()
                self.root.destroy()
            except tk.TclError:
                # GUI already destroyed, ignore the error
                pass


def create_thread_coordinator(thread_manager: ThreadManager) -> ThreadCoordinator:
    """
    Create and return a thread coordinator instance.
    
    Args:
        thread_manager: The ThreadManager instance to coordinate
        
    Returns:
        ThreadCoordinator instance
    """
    return ThreadCoordinator(thread_manager)
