import sqlite3
from datetime import datetime

def init_db(db_path='pipeline_logs.db'):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS pipeline_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            input_type TEXT,
            lhem_time REAL,
            tee_time REAL,
            del_time REAL,
            tcen_time REAL,
            cdf_time REAL,
            sra_time REAL,
            plugins_time REAL,
            total_time REAL,
            memory_rss_mb INTEGER,
            cpu_percent REAL,
            optimized_decision TEXT,
            plugin_outputs TEXT,
            error TEXT
        )
    ''')
    conn.commit()
    conn.close()

def log_pipeline(db_path, input_type, timings, resource, optimized_decision, plugin_outputs, error=None):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('''
        INSERT INTO pipeline_logs (
            input_type, lhem_time, tee_time, del_time, tcen_time, cdf_time, sra_time, plugins_time, total_time,
            memory_rss_mb, cpu_percent, optimized_decision, plugin_outputs, error
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        input_type,
        timings.get('lhem'), timings.get('tee'), timings.get('del'), timings.get('tcen'), timings.get('cdf'), timings.get('sra'), timings.get('plugins'), timings.get('total'),
        resource.get('memory_rss_mb'), resource.get('cpu_percent'),
        optimized_decision, str(plugin_outputs), error
    ))
    conn.commit()
    conn.close()
