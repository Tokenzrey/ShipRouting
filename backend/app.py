# app.py
import os
import json
import logging
import asyncio
from datetime import datetime, timedelta
from typing import List, Optional, Tuple, Dict

import aiofiles
from fastapi import FastAPI, HTTPException, Request, Depends, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from filelock import FileLock
import numpy as np

from controllers import get_wave_data_controller, djikstra_route_controller  # Sesuaikan dengan FastAPI
from managers import RouteOptimizer, fetch_and_cache_wave_data  # Pastikan ini diimplementasikan
from utils import WaveDataLocator, GridLocator, local_file_exists_for_all
from constants import (
    DATA_DIR_HTSGWSFC,
    DATA_DIR_DIRPWSFC,
    DATA_DIR_PERPWSFC,
    DATA_DIR_CACHE,
    DATA_DIR,
    Config
)

# Inisialisasi logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger(__name__)

# Buat direktori jika belum ada
for directory in [DATA_DIR, DATA_DIR_HTSGWSFC, DATA_DIR_DIRPWSFC, DATA_DIR_PERPWSFC, DATA_DIR_CACHE]:
    os.makedirs(directory, exist_ok=True)

# Inisialisasi FastAPI
app = FastAPI()

# Tambahkan middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Sesuaikan dengan kebutuhan keamanan Anda
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
grid_locator: Optional[GridLocator] = None
wave_data_locator: Optional[WaveDataLocator] = None
route_optimizer: Optional[RouteOptimizer] = None

# Variabel untuk melacak data gelombang terbaru
last_wave_file: Optional[str] = None
last_wave_file_mtime: Optional[float] = None

# Event untuk menandai selesainya inisialisasi
initialization_complete = asyncio.Event()

@app.on_event("startup")
async def startup_event():
    """
    Event startup untuk menginisialisasi global instances dan scheduler.
    """
    asyncio.create_task(initialize_global_instances())
    asyncio.create_task(initialize_scheduler())
    await initialization_complete.wait()

async def initialize_global_instances():
    global grid_locator, wave_data_locator, route_optimizer, last_wave_file, last_wave_file_mtime

    try:
        logger.info("Initializing WaveDataLocator...")
        wave_files = [f for f in os.listdir(DATA_DIR_CACHE) if f.endswith('.json')]
        if not wave_files:
            logger.warning("No wave data files found in the cache directory.")

        latest_wave_file = max(
            wave_files, key=lambda f: os.path.getmtime(os.path.join(DATA_DIR_CACHE, f))
        ) if wave_files else None
        latest_wave_file_path = os.path.join(DATA_DIR_CACHE, latest_wave_file) if latest_wave_file else None
        logger.info(f"Loading the latest wave data from {latest_wave_file}..." if latest_wave_file else "No latest wave data found.")

        if latest_wave_file:
            async with aiofiles.open(latest_wave_file_path, 'r') as f:
                wave_data_content = await f.read()
                wave_data = json.loads(wave_data_content)
                wave_data_locator = WaveDataLocator(wave_data, latest_wave_file)
        else:
            wave_data_locator = None

        logger.info("Initializing RouteOptimizer...")
        route_optimizer = RouteOptimizer(
            graph_file=Config.GRAPH_FILE,
            wave_data_locator=wave_data_locator,
            model_path=Config.MODEL_PATH,
            input_scaler_pkl=Config.INPUT_SCALER,
            output_scaler_pkl=Config.OUTPUT_SCALER,
            grid_locator=None  # Akan diatur setelah GridLocator diinisialisasi
        )

        logger.info("Initializing GridLocator...")
        if route_optimizer and route_optimizer.igraph_graph:
            # Mengambil koordinat dari igraph.Graph
            graph_coords = np.array([[v["lon"], v["lat"]] for v in route_optimizer.igraph_graph.vs])
            grid_locator = GridLocator(graph_coords)
            route_optimizer.grid_locator = grid_locator
            logger.info("GridLocator initialized and assigned to RouteOptimizer.")
        else:
            logger.warning("RouteOptimizer or its graph is not initialized properly.")
            grid_locator = None

        last_wave_file = latest_wave_file
        last_wave_file_mtime = os.path.getmtime(latest_wave_file_path) if latest_wave_file else None

        logger.info("Global instances initialized successfully.")

    except Exception as e:
        logger.error(f"Error during global instance initialization: {e}")
        wave_data_locator = None
        route_optimizer = None
        grid_locator = None

    finally:
        initialization_complete.set()

async def check_and_fetch_wave_data():
    logger.info("Starting scheduled task: check_and_fetch_wave_data")
    try:
        today = datetime.utcnow()
        for delta in range(0, 7):
            target_date = today - timedelta(days=delta)
            date_str = target_date.strftime("%Y%m%d")
            for time_slot in Config.TIME_SLOTS:
                if not local_file_exists_for_all(date_str, time_slot):
                    logger.info(f"Missing data for {date_str} - {time_slot}. Fetching...")
                    await asyncio.to_thread(fetch_and_cache_wave_data, date_str, time_slot)
                else:
                    logger.debug(f"Data for {date_str} - {time_slot} already exists.")
        logger.info("Scheduled task completed: check_and_fetch_wave_data")
    except Exception as e:
        logger.error(f"Error in scheduled task check_and_fetch_wave_data: {e}", exc_info=True)

async def initialize_scheduler():
    scheduler = AsyncIOScheduler()

    # Tambahkan job untuk menjalankan setiap hari pada pukul 00:00
    trigger = CronTrigger(hour=0, minute=0)
    scheduler.add_job(
        func=check_and_fetch_wave_data,
        trigger=trigger,
        id='check_and_fetch_wave_data',
        name='Check and fetch wave data daily',
        replace_existing=True
    )

    # Tambahkan job untuk menjalankan sekali saat startup
    scheduler.add_job(
        func=check_and_fetch_wave_data,
        trigger='date',
        run_date=datetime.utcnow(),
        id='initial_check_and_fetch_wave_data',
        name='Initial check and fetch wave data',
        replace_existing=True
    )

    scheduler.start()
    logger.info("APScheduler started and jobs added.")
    return scheduler

@app.middleware("http")
async def refresh_wave_data_locator(request: Request, call_next):
    """
    Middleware untuk memeriksa apakah data gelombang telah diperbarui sebelum memproses request.
    """
    global wave_data_locator, last_wave_file, last_wave_file_mtime, route_optimizer

    try:
        wave_files = [f for f in os.listdir(DATA_DIR_CACHE) if f.endswith('.json')]
        if not wave_files:
            logger.error("No wave data files found in the cache directory.")
            response = JSONResponse(
                status_code=503,
                content={"success": False, "error": "Wave data not available."}
            )
            return response

        latest_wave_file = max(
            wave_files, key=lambda f: os.path.getmtime(os.path.join(DATA_DIR_CACHE, f))
        )
        latest_wave_file_path = os.path.join(DATA_DIR_CACHE, latest_wave_file)
        latest_wave_file_mtime_new = os.path.getmtime(latest_wave_file_path)

        if last_wave_file == latest_wave_file and last_wave_file_mtime == latest_wave_file_mtime_new:
            logger.debug("Wave data has not changed. No update required.")
        else:
            logger.info(f"Loading updated wave data from {latest_wave_file}...")
            async with aiofiles.open(latest_wave_file_path, 'r') as f:
                wave_data_content = await f.read()
                wave_data = json.loads(wave_data_content)
                wave_data_locator = WaveDataLocator(wave_data, latest_wave_file)

                if route_optimizer:
                    route_optimizer.update_wave_data_locator(wave_data_locator)
                    logger.info("RouteOptimizer has been updated with the new WaveDataLocator.")

            last_wave_file = latest_wave_file
            last_wave_file_mtime = latest_wave_file_mtime_new

    except Exception as e:
        logger.error(f"Error while updating WaveDataLocator: {e}")

    response = await call_next(request)
    return response

@app.get("/api/wave_data")
async def api_get_wave_data(
    date: str = Query(None, description="Tanggal dalam format YYYYMMDD"),
    time_slot: str = Query(None, description="Slot waktu (00, 06, 12, 18)"),
    currentdate: bool = Query(True, description="Apakah menggunakan tanggal saat ini")
):
    """
    Endpoint untuk mendapatkan data gelombang.
    """
    if not initialization_complete.is_set():
        logger.info("Service is still initializing. Cannot process /api/wave_data request.")
        raise HTTPException(status_code=503, detail="Service is initializing. Please try again later.")

    if not route_optimizer:
        logger.info("RouteOptimizer or WaveDataLocator is not available.")
        raise HTTPException(status_code=503, detail="Service is not ready. Please try again later.")

    try:
        # Kirim parameter ke controller
        result = await asyncio.to_thread(
            get_wave_data_controller,
            route_optimizer,
            date,
            time_slot,
            currentdate
        )

        # Kembalikan data inti langsung
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error in /api/wave_data: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error.")

@app.post("/api/djikstra")
async def api_djikstra_route(request: Request):
    """
    Endpoint untuk menemukan rute menggunakan algoritma Dijkstra.
    """
    if not initialization_complete.is_set():
        logger.info("Service is still initializing. Cannot process /api/djikstra request.")
        raise HTTPException(status_code=503, detail="Service is initializing. Please try again later.")

    if not route_optimizer or not grid_locator:
        logger.info("RouteOptimizer or GridLocator is not available.")
        raise HTTPException(status_code=503, detail="Service is not ready. Please try again later.")

    try:
        payload = await request.json()
        result = await asyncio.to_thread(
            djikstra_route_controller,
            grid_locator,
            route_optimizer,
            payload
        )
        return JSONResponse(content=result)
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error in /api/djikstra: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error.")
