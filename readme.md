# Ship Routing with Dijkstra ANN and Non-ANN

Welcome to the Ship Routing project! This application leverages both ANN (Artificial Neural Network) and Non-ANN approaches for optimized ship routing, using a combination of a Next.js frontend and a Flask API backend.

---

## 🚀 Features

- Dynamic ship route optimization using Dijkstra algorithm.
- Integration of ANN for advanced routing under varying conditions.
- Wave data-based path calculations.
- Interactive user interface for route planning.

---

## 🛠️ Installation Guide

### Prerequisites

- **Operating System**: Windows, macOS, or Linux.
- **Python**: Version 3.10
- **Node.js**: Version 18 or above
- **npm**: Version 8 or above
- **pnpm**: Version 8 or above

---

### Step 1: Install Required Tools

#### Install Python 3.10

1. Download Python 3.10 from the [official website](https://www.python.org/downloads/release/python-3100/).
2. Follow the installation instructions for your operating system.
3. Ensure `python` and `pip` are accessible via the command line:
   ```bash
   python --version
   pip --version
   ```
4. **If `python --version` does not show 3.10**, set Python 3.10 as the default:
   - **Windows**:
     1. Open the Environment Variables settings.
     2. Add or update the `Path` variable to include the path to Python 3.10.
   - **macOS/Linux**:
     ```bash
     alias python=/path/to/python3.10
     ```
     Replace `/path/to/python3.10` with the actual path.

#### Install Node.js and npm

1. Download Node.js from the [official website](https://nodejs.org/).
2. Follow the installation instructions for your operating system.
3. Verify installation:
   ```bash
   node --version
   npm --version
   ```

#### Install pnpm

1. Install pnpm globally using npm:
   ```bash
   npm install -g pnpm
   ```
2. Verify installation:
   ```bash
   pnpm --version
   ```

---

### Step 2: Clone the Repository

1. Clone the repository to your local machine:
   ```bash
   git clone https://github.com/Tokenzrey/ShipRouting.git
   ```
2. Navigate to the project root:
   ```bash
   cd ship-routing
   ```

---

### Step 3: Download Region Graph File

1. Download the `region_graph.json` file from the following link:
   [Download region_graph.json](https://drive.google.com/file/d/1-5DmvwU_vOAAr79p3Jg9DKkgiIBWhxnT/view?usp=sharing)
2. Place the file in the `backend` folder:
   ```bash
   mv region_graph.pkl backend/
   ```

---

### Step 4: Backend Setup

1. Navigate to the backend folder:
   ```bash
   cd backend
   ```
2. Create a Python virtual environment:
   ```bash
   python -m venv venv
   ```
3. Activate the virtual environment:
   - **Windows**:
     ```bash
     venv\Scripts\activate
     ```
   - **macOS/Linux**:
     ```bash
     source venv/bin/activate
     ```
4. Install required Python packages:
   ```bash
   pip install -r requirements.txt
   ```
5. Run the Flask API:
   ```bash
   uvicorn app:app --host 0.0.0.0 --port 5000 --reload
   ```

---

### Step 5: Frontend Setup

1. Navigate to the frontend folder:
   ```bash
   cd ../frontend
   ```
2. Install dependencies using pnpm:
   ```bash
   pnpm install
   ```
3. Start the development server:
   ```bash
   pnpm dev
   ```

---

## 🌐 Access the Application

1. Open your web browser and navigate to:
   ```
   http://localhost:3000
   ```
2. The backend API is accessible at:
   ```
   http://localhost:5000/api
   ```

---

## 📂 Project Structure

```
ship-routing/
├── backend/        # Flask API backend
├── frontend/       # Next.js frontend
└── README.md       # Project documentation
```

---

## 🤝 Contribution

We welcome contributions to improve the Ship Routing project. Please submit issues or pull requests on the [GitHub repository](https://github.com/your-repository/ship-routing).

---

## 📄 License

This project is licensed under the MIT License. See the LICENSE file for details.

---

Happy coding! 🚢
