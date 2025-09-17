// src/pages/Events.tsx
import React, { useEffect, useState } from "react";
import axiosClient from "../api/axiosClient";
import { Link } from "react-router-dom";

interface EventItem {
  _id: string;
  title: string;
  description: string;
  date: string;
  createdBy?: { name?: string; email?: string };
}

const Events: React.FC = () => {
  const [events, setEvents] = useState<EventItem[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetch = async () => {
      try {
        const res = await axiosClient.get("/events");
        setEvents(res.data);
      } catch (err) {
        console.error(err);
      } finally {
        setLoading(false);
      }
    };
    fetch();
  }, []);

  if (loading)
    return (
      <div className="p-6 text-white text-center text-lg animate-pulse">
        Loading events...
      </div>
    );

  return (
    <div className="p-6 min-h-screen bg-gradient-to-br from-gray-900 to-black text-white">
      <h1 className="text-4xl font-bold mb-10 text-center text-purple-400 drop-shadow-md">
        Upcoming Events
      </h1>

      <div className="grid gap-8 grid-cols-1 sm:grid-cols-2 lg:grid-cols-3">
        {events.map((ev, index) => (
          <div
            key={ev._id}
            className={`bg-gray-800 border border-gray-700 p-6 rounded-xl shadow-lg transform transition duration-500 ease-in-out hover:scale-105 hover:shadow-purple-500/30 
            ${index % 2 === 0 ? "animate-fade-in-up delay-100" : "animate-fade-in-up delay-200"}`}
          >
            <h2 className="text-2xl font-semibold text-purple-300 mb-2">
              {ev.title}
            </h2>
            <p className="text-sm text-gray-300 mb-4">
              {ev.description.slice(0, 120)}
              {ev.description.length > 120 ? "..." : ""}
            </p>
            <p className="text-xs text-gray-400 mb-4">
              By <span className="font-medium">{ev.createdBy?.name || "Unknown"}</span>{" "}
              — {new Date(ev.date).toLocaleDateString()}
            </p>

            <Link
              to={`/events/${ev._id}`}
              className="inline-block px-4 py-2 rounded-md bg-purple-600 hover:bg-purple-700 transition-colors duration-300 text-white font-semibold text-sm"
            >
              View Details
            </Link>
          </div>
        ))}
      </div>
    </div>
  );
};

export default Events;
