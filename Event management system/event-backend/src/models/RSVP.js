// src/models/RSVP.js
import mongoose from "mongoose";

const rsvpSchema = new mongoose.Schema(
  {
    event: {
      type: mongoose.Schema.Types.ObjectId,
      ref: "Event",
      required: true,
    },
    user: {
      type: mongoose.Schema.Types.ObjectId,
      ref: "User",
      required: true,
    },
    status: {
      type: String,
      enum: ["attending", "cancelled"],
      default: "attending",
    },
  },
  { timestamps: true }
);

// Ensure a user can RSVP only once per event
rsvpSchema.index({ event: 1, user: 1 }, { unique: true });

export default mongoose.model("RSVP", rsvpSchema);
