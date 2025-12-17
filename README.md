## Financial Forecasting LSTM Web Application

### Project Demo Video

Click the link to watch the full demo -->
[Object Detection Demo](https://abumusalab.com.ng/financial-forecasting-app/)


**Project Overview**
This project delivers a professional-grade, interactive web application for financial forecasting using Long Short-Term Memory (LSTM) neural networks. 
This application showcases advanced machine learning techniques, real-time data processing, and robust full-stack development practices. 

**Key Features**
- LSTM Neural Networks: Utilizes advanced LSTM models for accurate time series forecasting of stock prices.
- Real-time Data Ingestion: Integrates real-time market data streaming via WebSockets for up-to-date predictions.
- Uncertainty Quantification: Provides confidence intervals alongside predictions, offering a more comprehensive view of forecast reliability.
- Interactive Dashboard: A user-friendly React-based frontend with dynamic charts and controls for exploring historical data, training models, and generating predictions.
- Model Evaluation: Allows users to evaluate trained models with key metrics (RMSE, MAE, MAPE) directly within the dashboard.


**Technologies Used**

**Backend**:
- **Python:** Core programming language.
- **Flask:** Lightweight web framework for API and WebSocket handling.
- **TensorFlow/Keras:** For building and training LSTM models.
- **Scikit-learn:** For data preprocessing and scaling.
- **Flask-SocketIO:** For real-time bidirectional communication.
- **SQLAlchemy & SQLite:** For database management and persistence.

**Frontend**:
- **React:** JavaScript library for building interactive user interfaces.
- **Vite:** Fast build tool for modern web projects.
- **Recharts:** Composable charting library for React.
- **Tailwind CSS & shadcn/ui:** For responsive and aesthetically pleasing UI components.

**Deployment**:
- **Docker:** Containerization for consistent environments.
- **Docker Compose:** For defining and running multi-container Docker applications.
- **AWS (Amazon Web Services)**

**Getting Started**

Follow these instructions to set up and run the application locally.
Prerequisites
- **Python 3.8+**
- **Node.js 18+**
- **pnpm (recommended for frontend dependencies)**
- **Docker (optional, for containerized local testing)**

Local Setup (Recommended for Development)
1. Clone the repository:
2. Backend Setup:
3. Frontend Setup:
4. Access the Application:
Open your web browser and navigate to http://localhost:5173.

Local Setup (Using Docker Compose)
If you have Docker installed, you can run the entire application using Docker Compose, which mimics the production environment:
1. Navigate to the main project directory:
2. Build and run the Docker containers:





