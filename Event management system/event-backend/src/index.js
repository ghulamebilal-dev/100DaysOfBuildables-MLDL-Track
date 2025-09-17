// src/index.js
// Entry point for the backend server (ES Modules / import syntax)

import express from "express";
import morgan from "morgan";
import cors from "cors";
import dotenv from "dotenv";
import connectDB from "./config/db.js";
import healthRoutes from "./routes/health.js";
import authRoutes from "./routes/auth.js";
import eventRoutes from "./routes/event.js";
import rsvpRoutes from "./routes/rsvp.js";
dotenv.config(); // loads variables from .env into process.env

const app = express();

// Middleware
app.use(cors()); // enable CORS
app.use(morgan("dev")); // simple request logging
app.use(express.json()); // parse JSON bodies

// Routes
app.use("/api/health", healthRoutes);
app.use("/api/auth", authRoutes);
app.use("/api/events", eventRoutes);
app.use("/api/rsvp", rsvpRoutes);



// Start server after DB connection
const PORT = process.env.PORT || 5000;
const startServer = async () => {
  try {
    // connect to MongoDB using connection string from environment
    await connectDB(process.env.MONGO_URI);

    app.listen(PORT, () => {
      console.log(`✅ Server running on http://localhost:${PORT}`);
    });
  } catch (err) {
    console.error("Failed to start server:", err);
    process.exit(1);
  }
};

startServer();
