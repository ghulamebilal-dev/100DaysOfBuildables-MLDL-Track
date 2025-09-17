import React, { useEffect, useState, useCallback } from "react";
import { useParams, Link } from "react-router-dom";
import axiosClient from "../api/axiosClient";
import { useAuth } from "../context/useAuth";
import axios from "axios";

interface EventType {
  _id: string;
  title: string;
  description: string;
  date: string;
  createdBy: { _id?: string; name?: string; email?: string };
}

interface Attendee {
  _id: string;
  user: { _id?: string; name?: string; email?: string };
  status: string;
}

const EventDetails: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const { isAuthenticated, user } = useAuth();

  const [event, setEvent] = useState<EventType | null>(null);
  const [attendees, setAttendees] = useState<Attendee[]>([]);
  const [loading, setLoading] = useState(true);
  const [msg, setMsg] = useState("");
  const [isAttending, setIsAttending] = useState(false);

  const fetchEventData = useCallback(async () => {
    if (!id) return;
    try {
      const [evRes, atRes] = await Promise.all([
        axiosClient.get(`/events/${id}`),
        axiosClient.get(`/rsvp/${id}`),
      ]);
      setEvent(evRes.data);
      setAttendees(atRes.data);

      if (user) {
        const attending = atRes.data.some(
          (a: Attendee) =>
            a.user._id === user.id && a.status === "attending"
        );
        setIsAttending(attending);
      }
    } catch (error) {
      console.error(error);
    } finally {
      setLoading(false);
    }
  }, [id, user]);

  useEffect(() => {
    fetchEventData();
  }, [fetchEventData]);

  const handleRSVP = async () => {
    try {
      await axiosClient.post(`/rsvp/${id}`);
      setMsg("🎉 RSVP successful");
      await fetchEventData();
      setTimeout(() => setMsg(""), 2000);
    } catch (error: unknown) {
      if (axios.isAxiosError(error)) {
        setMsg(error.response?.data?.message || "Failed to RSVP");
      } else if (error instanceof Error) {
        setMsg(error.message);
      } else {
        setMsg("Failed to RSVP");
      }
      setTimeout(() => setMsg(""), 2000);
    }
  };

  const handleCancel = async () => {
    try {
      await axiosClient.delete(`/rsvp/${id}`);
      setMsg("❌ RSVP cancelled");
      await fetchEventData();
      setTimeout(() => setMsg(""), 2000);
    } catch (error: unknown) {
      if (axios.isAxiosError(error)) {
        setMsg(error.response?.data?.message || "Failed to cancel RSVP");
      } else if (error instanceof Error) {
        setMsg(error.message);
      } else {
        setMsg("Failed to cancel RSVP");
      }
      setTimeout(() => setMsg(""), 2000);
    }
  };

  if (loading)
    return <div className="p-6 text-white text-center animate-pulse">Loading...</div>;
  if (!event)
    return <div className="p-6 text-red-500 text-center">Event not found</div>;

  return (
    <div className="bg-gray-950 text-gray-100 min-h-screen p-6">
      <div className="max-w-3xl mx-auto space-y-8">

        {/* Event Card */}
        <div className="bg-gray-900/95 backdrop-blur-sm p-6 rounded-xl shadow-md animate-fade-in-up transition-all duration-700">
          <h1 className="text-3xl font-bold text-purple-500">{event.title}</h1>
          <p className="text-sm text-gray-400 mt-1">
            By <span className="font-medium">{event.createdBy?.name || "Unknown"}</span> on{" "}
            {new Date(event.date).toLocaleString()}
          </p>
          <p className="mt-4 text-gray-300 leading-relaxed">
            {event.description}
          </p>

          {/* Success/Error Message */}
          {msg && (
            <div className="mt-4 p-3 bg-purple-700/90 text-white rounded-md animate-fade-in">
              {msg}
            </div>
          )}

          {/* RSVP Buttons */}
          <div className="mt-6 flex flex-wrap gap-3">
            {isAuthenticated ? (
              <>
                <button
                  onClick={handleRSVP}
                  disabled={isAttending}
                  className={`px-4 py-2 rounded font-semibold transition duration-300 ${
                    isAttending
                      ? "bg-gray-500 text-white cursor-not-allowed"
                      : "bg-purple-600 hover:bg-purple-700 text-white"
                  }`}
                >
                  {isAttending ? "Already RSVP’d" : "RSVP Now"}
                </button>
                {isAttending && (
                 <button
  onClick={handleCancel}
  className="px-4 py-2 rounded bg-purple-700 hover:bg-purple-800 text-white transition duration-300"
>
                    Cancel RSVP
                  </button>
                )}
              </>
            ) : (
              <Link
                to="/login"
                className="px-4 py-2 bg-purple-600 hover:bg-purple-700 text-white rounded transition"
              >
                Login to RSVP
              </Link>
            )}
          </div>
        </div>

        {/* Attendees */}
        <div className="bg-gray-900/95 backdrop-blur-sm p-6 rounded-xl shadow-md animate-fade-in-up">
          <h3 className="text-xl font-semibold text-purple-500 mb-4">
            Attendees ({attendees.length})
          </h3>
          <ul className="space-y-2">
            {attendees.length === 0 ? (
              <li className="text-sm text-gray-500">No attendees yet.</li>
            ) : (
              attendees.map((a) => (
                <li
                  key={a._id}
                  className="border-b border-gray-700 pb-2 text-sm text-gray-300"
                >
                  <span className="font-medium">{a.user.name}</span>{" "}
                  <span className="text-xs text-gray-500">
                    ({a.user.email})
                  </span>
                </li>
              ))
            )}
          </ul>
        </div>
      </div>
    </div>
  );
};

export default EventDetails;
