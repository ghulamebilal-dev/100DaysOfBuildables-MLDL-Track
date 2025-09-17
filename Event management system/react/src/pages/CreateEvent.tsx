import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import axiosClient from "../api/axiosClient";
import { useAuth } from "../context/useAuth";
import axios from "axios";
import "../styles/CreateEvent.css"; // 🔥 Custom animations & theme styles

const CreateEvent: React.FC = () => {
  const { isAuthenticated } = useAuth();
  const navigate = useNavigate();

  const [title, setTitle] = useState("");
  const [description, setDescription] = useState("");
  const [date, setDate] = useState("");
  const [err, setErr] = useState("");
  const [loading, setLoading] = useState(false);

  if (!isAuthenticated) {
    navigate("/login");
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setErr("");
    setLoading(true);

    try {
      await axiosClient.post("/events", { title, description, date });
      navigate("/events");
    } catch (error: unknown) {
      if (axios.isAxiosError(error)) {
        setErr(error.response?.data?.message || "Failed to create event");
      } else if (error instanceof Error) {
        setErr(error.message);
      } else {
        setErr("Failed to create event");
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-[70vh] flex items-center justify-center p-4 bg-[#0f0e17] text-white fade-in">
      <form
        onSubmit={handleSubmit}
        className="w-full max-w-md bg-[#1a1823] border border-purple-700 p-6 rounded-xl shadow-lg fade-in-up"
      >
        <h2 className="text-3xl font-bold mb-6 text-center text-purple-400">
          Create Event
        </h2>

        {err && <div className="mb-4 text-red-500">{err}</div>}

        <label className="block mb-2 font-medium" htmlFor="title">
          Event Title
        </label>
        <input
          id="title"
          value={title}
          onChange={(e) => setTitle(e.target.value)}
          placeholder="Event Title"
          className="w-full mb-4 p-2 rounded bg-gray-800 border border-purple-500 text-white focus:ring-2 focus:ring-purple-600"
          required
        />

        <label className="block mb-2 font-medium" htmlFor="description">
          Description
        </label>
        <textarea
          id="description"
          value={description}
          onChange={(e) => setDescription(e.target.value)}
          placeholder="Event Description"
          className="w-full mb-4 p-2 rounded bg-gray-800 border border-purple-500 text-white focus:ring-2 focus:ring-purple-600"
          required
        />

        <label className="block mb-2 font-medium" htmlFor="date">
          Event Date
        </label>
        <input
          id="date"
          type="date"
          value={date}
          onChange={(e) => setDate(e.target.value)}
          className="w-full mb-6 p-2 rounded bg-gray-800 border border-purple-500 text-white focus:ring-2 focus:ring-purple-600"
          required
        />

        <button
          type="submit"
          disabled={loading}
          className="w-full bg-purple-700 hover:bg-purple-800 text-white py-2 rounded transition duration-300 disabled:opacity-50"
        >
          {loading ? "Creating..." : "Create Event"}
        </button>
      </form>
    </div>
  );
};

export default CreateEvent;
