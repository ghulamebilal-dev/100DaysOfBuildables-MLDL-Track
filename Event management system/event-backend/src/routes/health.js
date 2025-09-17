// src/routes/health.js
import express from "express";

const router = express.Router();

/**
 * GET /api/health
 * Simple health check for uptime & quick test.
 */
router.get("/", (req, res) => {
  res.json({
    status: "ok",
    uptime: process.uptime(),
    timestamp: new Date().toISOString(),
  });
});

export default router;
