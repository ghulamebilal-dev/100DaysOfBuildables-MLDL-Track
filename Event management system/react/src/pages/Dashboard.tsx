// src/pages/Dashboard.tsx
import React, { useEffect, useState, useCallback } from "react";
import axiosClient from "../api/axiosClient";
import { useAuth } from "../context/useAuth";
import axios from "axios";
import "../styles/Dashboard.css"; // ✅ Custom styles with animations and color overrides

interface EventType {
  _id: string;
  title: string;
  description: string;
  date: string;
  createdBy: { name: string };
}

const Dashboard: React.FC = () => {
  const { user } = useAuth();
  const [events, setEvents] = useState<EventType[]>([]);
  const [loading, setLoading] = useState(true);
  const [msg, setMsg] = useState("");

  const [editingEvent, setEditingEvent] = useState<EventType | null>(null);
  const [editTitle, setEditTitle] = useState("");
  const [editDescription, setEditDescription] = useState("");
  const [editDate, setEditDate] = useState("");

  const fetchEvents = useCallback(async () => {
    try {
      const res = await axiosClient.get("/events");
      const myEvents = res.data.filter(
        (e: EventType) => e.createdBy?.name === user?.name
      );
      setEvents(myEvents);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  }, [user?.name]);

  useEffect(() => {
    fetchEvents();
  }, [fetchEvents]);

  const handleDelete = async (id: string) => {
    try {
      await axiosClient.delete(`/events/${id}`);
      setMsg("Event deleted");
      fetchEvents();
      setTimeout(() => setMsg(""), 2000);
    } catch (error: unknown) {
      if (axios.isAxiosError(error)) {
        setMsg(error.response?.data?.message || "Failed to delete");
      } else if (error instanceof Error) {
        setMsg(error.message);
      } else {
        setMsg("Failed to delete");
      }
      setTimeout(() => setMsg(""), 2000);
    }
  };

  const openEditModal = (event: EventType) => {
    setEditingEvent(event);
    setEditTitle(event.title);
    setEditDescription(event.description);
    setEditDate(new Date(event.date).toISOString().slice(0, 16));
  };

  const handleEditSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!editingEvent) return;

    try {
      await axiosClient.put(`/events/${editingEvent._id}`, {
        title: editTitle,
        description: editDescription,
        date: editDate,
      });
      setMsg("Event updated successfully 🎉");
      setEditingEvent(null);
      fetchEvents();
      setTimeout(() => setMsg(""), 2000);
    } catch (error: unknown) {
      if (axios.isAxiosError(error)) {
        setMsg(error.response?.data?.message || "Failed to update");
      } else if (error instanceof Error) {
        setMsg(error.message);
      } else {
        setMsg("Failed to update");
      }
      setTimeout(() => setMsg(""), 2000);
    }
  };

  if (loading) return <div className="p-6">Loading...</div>;

  return (
    <div className="p-6 fade-in bg-[#0f0e17] min-h-screen text-white">
      <h1 className="text-3xl font-bold mb-6 text-purple-400">My Events</h1>

      {msg && (
        <div className="mb-4 p-3 bg-purple-600 text-white rounded fade-in">
          {msg}
        </div>
      )}

      {events.length === 0 && <p>You haven’t created any events yet.</p>}

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {events.map((event) => (
          <div
            key={event._id}
            className="bg-[#1a1823] p-5 rounded-xl shadow-md transition-all fade-in-up border border-purple-700"
          >
            <h2 className="text-xl font-semibold text-purple-300">
              {event.title}
            </h2>
            <p className="text-sm text-gray-400">
              {new Date(event.date).toLocaleString()}
            </p>
            <p className="mt-3 text-gray-300">{event.description}</p>
            <div className="mt-4 flex gap-3">
              <button
                onClick={() => openEditModal(event)}
                className="px-4 py-1 rounded bg-purple-700 hover:bg-purple-800 text-white transition hover:scale-105"
              >
                Edit
              </button>
              <button
                onClick={() => handleDelete(event._id)}
                className="px-3 py-1 bg-purple-700 hover:bg-purple-800 text-white rounded transition-colors duration-200"
              >
                Delete
              </button>
            </div>
          </div>
        ))}
      </div>

      {/* Edit Modal */}
      {editingEvent && (
        <div className="fixed inset-0 flex items-center justify-center bg-black/70 z-50 fade-in">
          <div className="bg-[#1a1823] text-white p-6 rounded-xl shadow-lg max-w-md w-full fade-in-up border border-purple-700">
            <h2 className="text-xl font-bold mb-4 text-purple-400">
              Edit Event
            </h2>
            <form onSubmit={handleEditSubmit} className="space-y-4">
              <div>
                <label htmlFor="editTitle" className="block text-sm font-medium">
                  Title
                </label>
                <input
                  id="editTitle"
                  type="text"
                  value={editTitle}
                  onChange={(e) => setEditTitle(e.target.value)}
                  className="w-full p-2 rounded bg-gray-800 border border-purple-500 text-white"
                  required
                />
              </div>

              <div>
                <label htmlFor="editDescription" className="block text-sm font-medium">
                  Description
                </label>
                <textarea
                  id="editDescription"
                  value={editDescription}
                  onChange={(e) => setEditDescription(e.target.value)}
                  className="w-full p-2 rounded bg-gray-800 border border-purple-500 text-white"
                  required
                />
              </div>

              <div>
                <label htmlFor="editDate" className="block text-sm font-medium">
                  Date
                </label>
                <input
                  id="editDate"
                  type="datetime-local"
                  value={editDate}
                  onChange={(e) => setEditDate(e.target.value)}
                  className="w-full p-2 rounded bg-gray-800 border border-purple-500 text-white"
                  required
                />
              </div>

              <div className="flex justify-end gap-3">
                <button
                  type="button"
                  onClick={() => setEditingEvent(null)}
                  className="px-4 py-1 border border-gray-400 hover:bg-gray-700 rounded transition"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  className="px-4 py-1 bg-purple-700 hover:bg-purple-800 text-white rounded transition"
                >
                  Save Changes
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
};

export default Dashboard;
