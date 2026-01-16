import threading
import time
import queue
from enum import Enum
from typing import Optional

class ThreadState(Enum):
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    STARTED = "started"

class ThreadManager:
    def __init__(self):
        self.thread_a_state = ThreadState.STOPPED
        self.thread_b_state = ThreadState.STOPPED
        
        self.threada: Optional[threading.Thread] = None
        self.threadb: Optional[threading.Thread] = None
        
        self.running_a = threading.Event()
        self.running_b = threading.Event()
        self.paused_a = threading.Event()
        self.started_a = threading.Event()
        self.paused_b = threading.Event()
        self.stopped_a = threading.Event()
        self.stopped_b = threading.Event()
        
        self.state_lock = threading.Lock()
        
        self.thread_a_resume_allowed = True
        self.thread_b_resume_allowed = False
        
    def start_thread_a(self):
        with self.state_lock:
            if self.thread_a_state == ThreadState.STOPPED or self.thread_a_state == ThreadState.PAUSED:
                self.thread_a_state = ThreadState.RUNNING
                self.running_a.set()
                self.paused_a.clear()
                self.stopped_a.clear()
                self.started_a.set()
                
    def start_thread_b(self):
        with self.state_lock:
            if self.thread_b_state == ThreadState.STOPPED or self.thread_b_state == ThreadState.PAUSED:
                self.thread_b_state = ThreadState.RUNNING
                self.running_b.set()
                self.paused_b.clear()
                self.stopped_b.clear()
                
    def pause_thread_a(self):
        with self.state_lock:
            if self.thread_a_state == ThreadState.RUNNING:
                self.thread_a_state = ThreadState.PAUSED
                self.running_a.clear()
                self.paused_a.set()
                self.stopped_a.clear()
                
    def pause_thread_b(self):
        with self.state_lock:
            if self.thread_b_state == ThreadState.RUNNING:
                self.thread_b_state = ThreadState.PAUSED
                self.running_b.clear()
                self.paused_b.set()
                self.stopped_b.clear()
                
    def stop_thread_a(self):
        with self.state_lock:
            if self.thread_a_state == ThreadState.RUNNING:
                self.thread_a_state = ThreadState.STOPPED
                self.running_a.clear()
                self.paused_a.clear()
                self.stopped_a.set()
                
    def stop_thread_b(self):
        with self.state_lock:
            if self.thread_b_state == ThreadState.RUNNING:
                self.thread_b_state = ThreadState.STOPPED
                self.running_b.clear()
                self.paused_b.clear()
                self.stopped_b.set()

    def stop_all_threads(self):
        with self.state_lock:
            self.stop_thread_a()
            self.stop_thread_b()