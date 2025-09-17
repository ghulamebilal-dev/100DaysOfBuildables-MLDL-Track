// src/config/db.js
// Centralized mongoose connection helper (ES Modules)

import mongoose from "mongoose";

/**
 * Connect to MongoDB using mongoose.
 * Throws if MONGO_URI is not provided or connection fails.
 * @param {string} uri - MongoDB connection string
 */
const connectDB = async (uri) => {
  if (!uri) {
    throw new Error("MONGO_URI is not set in environment variables.");
  }

  try {
    // mongoose connect options: modern mongoose doesn't need useCreateIndex, etc.
    await mongoose.connect(uri, {
      // these options are now defaults in newer mongoose versions, but harmless to include
      useNewUrlParser: true,
      useUnifiedTopology: true,
    });
    console.log("✅ MongoDB connected");
  } catch (err) {
    console.error("MongoDB connection error:", err);
    throw err;
  }
};

export default connectDB;
