// src/routes/rsvp.js
import express from "express";
import authMiddleware from "../middleware/authMiddleware.js";
import {
  attendEvent,
  cancelRSVP,
  getAttendees,
} from "../controllers/rsvpController.js";

const router = express.Router();

// RSVP to event (protected)
router.post("/:eventId", authMiddleware, attendEvent);

// Cancel RSVP (protected)
router.delete("/:eventId", authMiddleware, cancelRSVP);

// Get attendees for event (public)
router.get("/:eventId", getAttendees);

export default router;
