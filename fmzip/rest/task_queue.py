import time
from queue import Queue
from threading import Timer
from threading import Condition
from threading import Thread, Lock

class TaskQueueWithWatcher:
    def __init__(self, batch_size=4):
        self.buffer = Queue()
        self.batch_size = batch_size
        self.lock = Lock()
        self.results = {}
        self.condition = Condition(self.lock)
        self.processing_thread = Thread(target=self.process_tasks)
        self.processing_thread.daemon = True
        self.processing_thread.start()

    def add_task(self, task):
        with self.condition:
            self.buffer.put(task)
            self.condition.wait_for(lambda: task['id'] in self.results)
            return self.results.pop(task['id'])

    def process_tasks(self):
        while True:
            batch = []
            while not self.buffer.empty() and len(batch) < self.batch_size:
                task = self.buffer.get()
                batch.append(task)
            
            if len(batch) > 0:
                self.execute_batch(batch)
            
            time.sleep(1)

    def execute_batch(self, batch):
        with self.condition:
            print(f"Processing batch: {batch}")
            for task in batch:
                # Simulate task execution
                print(f"Executing task: {task}")
                time.sleep(0.2)
                self.results[task['id']] = task
            print("Batch processing completed.\n")
            self.condition.notify_all()  # Notify all waiting threads

class TaskQueueWithTimeout(TaskQueueWithWatcher):
    def __init__(self, batch_size=4, timeout=5):
        super().__init__(batch_size)
        self.timeout = timeout
        self.timer = None
        self.reset_timer()
        
    def reset_timer(self):
        if self.timer:
            self.timer.cancel()
        self.timer = Timer(self.timeout, self.process_timeout)
        self.timer.start()
        
    def add_task(self, task):
        with self.condition:
            self.buffer.put(task)
            self.reset_timer()  # Reset the timer upon adding a new task
            self.condition.wait_for(lambda: task['id'] in self.results)
            return self.results.pop(task['id'])
    def process_tasks(self):
        while True:
            batch = []
            while not self.buffer.empty() and len(batch) < self.batch_size:
                task = self.buffer.get()
                batch.append(task)
            
            if len(batch) > 0:
                self.execute_batch(batch)
                self.reset_timer()
            
            time.sleep(1)
            
    def process_timeout(self):
        with self.condition:
            batch = []
            while not self.buffer.empty() and len(batch) < self.batch_size:
                task = self.buffer.get()
                batch.append(task)
            
            if len(batch) > 0:
                self.execute_batch(batch)
            self.reset_timer()  # Reset the timer after timeout-triggered processing

# Initialize task queue with timeout
task_queue_timeout = TaskQueueWithTimeout(batch_size=3, timeout=6)

# Function to add and wait for tasks with timeout
def add_and_wait_for_tasks_with_timeout(task, idx):
    task['id'] = idx
    result = task_queue_timeout.add_task(task)
    print(f"Received result for {task}: {result}")

tasks = [
    {'prompt': "Once upon a time, ", 'model': 'gpt2'},
    {'prompt': "Alan Turing is ", 'model': 'gpt3'},
    {'prompt': "Computer Science is about ", 'model': 'gpt4'},
]

# Add tasks and wait for their completion
waiting_threads_timeout = [Thread(target=add_and_wait_for_tasks_with_timeout, args=(task, idx, )) for idx, task in enumerate(tasks)]
for thread in waiting_threads_timeout:
    thread.start()

# Allow some time for tasks to be processed and results to be returned
time.sleep(2)  # Waiting for 20 seconds to observe the timeout-triggered batch processing
