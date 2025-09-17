// src/controllers/rsvpController.js
import RSVP from "../models/RSVP.js";
import Event from "../models/Event.js";

/**
 * RSVP to an event
 * POST /api/rsvp/:eventId
 */
export const attendEvent = async (req, res) => {
  try {
    const { eventId } = req.params;

    // Check if event exists
    const event = await Event.findById(eventId);
    if (!event) return res.status(404).json({ message: "Event not found" });

    // Create or update RSVP
    const rsvp = await RSVP.findOneAndUpdate(
      { event: eventId, user: req.user._id },
      { status: "attending" },
      { upsert: true, new: true }
    );

    res.status(200).json({ message: "RSVP successful", rsvp });
  } catch (err) {
    console.error("RSVP error:", err);
    res.status(500).json({ message: "Server error" });
  }
};

/**
 * Cancel RSVP
 * DELETE /api/rsvp/:eventId
 */
export const cancelRSVP = async (req, res) => {
  try {
    const { eventId } = req.params;

    const rsvp = await RSVP.findOneAndUpdate(
      { event: eventId, user: req.user._id },
      { status: "cancelled" },
      { new: true }
    );

    if (!rsvp) return res.status(404).json({ message: "RSVP not found" });

    res.status(200).json({ message: "RSVP cancelled", rsvp });
  } catch (err) {
    console.error("Cancel RSVP error:", err);
    res.status(500).json({ message: "Server error" });
  }
};

/**
 * Get attendees of an event
 * GET /api/rsvp/:eventId
 */
export const getAttendees = async (req, res) => {
  try {
    const { eventId } = req.params;

    const attendees = await RSVP.find({ event: eventId, status: "attending" })
      .populate("user", "name email")
      .select("-__v");

    res.status(200).json(attendees);
  } catch (err) {
    console.error("Get attendees error:", err);
    res.status(500).json({ message: "Server error" });
  }
};
