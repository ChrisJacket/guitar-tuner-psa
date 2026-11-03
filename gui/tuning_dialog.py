"""
gui/tuning_dialog.py
====================
Modal dialog window with tabs for tuning categories and custom tuning creation.
"""
import tkinter as tk
from tkinter import messagebox, ttk
from typing import Callable, Optional
from tunings_manager import TuningsManager, infer_string_notes


class TuningDialog(tk.Toplevel):
    """Separate window containing category tabs and custom tuning builder."""

    def __init__(
        self,
        parent: tk.Tk,
        tunings_mgr: TuningsManager,
        on_applied: Callable[[str], None]
    ):
        super().__init__(parent)
        self.tunings_mgr = tunings_mgr
        self.on_applied = on_applied

        self.title("Select Guitar Tuning")
        self.geometry("640x520")
        self.minsize(560, 440)
        self.transient(parent)
        self.grab_set()

        self._build_ui()
        self.focus_set()

    def _build_ui(self):
        main_frame = ttk.Frame(self, padding=12)
        main_frame.pack(fill="both", expand=True)

        ttk.Label(
            main_frame,
            text="Choose a target tuning or create a custom one:",
            font=("Helvetica", 11, "bold")
        ).pack(anchor="w", pady=(0, 8))

        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill="both", expand=True)

        categories = self.tunings_mgr.get_categories()
        self.category_listboxes = {}

        # Add tabs for each existing category
        for cat_name, tunings in categories.items():
            tab = ttk.Frame(self.notebook, padding=10)
            self.notebook.add(tab, text=f" {cat_name} ")
            self._build_category_tab(tab, cat_name, tunings)

        # Add Custom Tuning tab
        custom_tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(custom_tab, text=" Custom Tuning ")
        self._build_custom_tab(custom_tab)

        # Bottom button bar
        btn_bar = ttk.Frame(main_frame, padding=(0, 10, 0, 0))
        btn_bar.pack(fill="x")
        ttk.Button(btn_bar, text="Close", command=self.destroy).pack(side="right", padx=4)

        # Switch to active tuning tab if possible
        active_cat = self.tunings_mgr.active_category
        for idx, tab_id in enumerate(self.notebook.tabs()):
            if active_cat in self.notebook.tab(tab_id, "text"):
                self.notebook.select(idx)
                break

    def _build_category_tab(self, parent: ttk.Frame, cat_name: str, tunings: dict):
        # Split into list and preview
        paned = ttk.PanedWindow(parent, orient="horizontal")
        paned.pack(fill="both", expand=True)

        list_frame = ttk.Frame(paned)
        paned.add(list_frame, weight=1)

        scrollbar = ttk.Scrollbar(list_frame, orient="vertical")
        listbox = tk.Listbox(
            list_frame,
            yscrollcommand=scrollbar.set,
            font=("Helvetica", 10),
            activestyle="none",
            selectmode="single"
        )
        scrollbar.config(command=listbox.yview)
        scrollbar.pack(side="right", fill="y")
        listbox.pack(side="left", fill="both", expand=True)

        names = list(tunings.keys())
        for idx, name in enumerate(names):
            listbox.insert("end", f"  {name}")
            if name == self.tunings_mgr.active_tuning_name:
                listbox.selection_set(idx)
                listbox.see(idx)

        # Right frame: details preview & Apply button
        detail_frame = ttk.Frame(paned, padding=(12, 4))
        paned.add(detail_frame, weight=1)

        title_lbl = ttk.Label(detail_frame, text="", font=("Helvetica", 11, "bold"))
        title_lbl.pack(anchor="w", pady=(0, 6))

        notes_raw_lbl = ttk.Label(detail_frame, text="", foreground="#666666")
        notes_raw_lbl.pack(anchor="w", pady=(0, 8))

        strings_box = ttk.LabelFrame(detail_frame, text="Inferred Strings & Frequencies", padding=8)
        strings_box.pack(fill="both", expand=True, pady=(0, 10))

        strings_lbl = ttk.Label(strings_box, text="", justify="left", font=("Helvetica", 10))
        strings_lbl.pack(anchor="nw", fill="both", expand=True)

        apply_btn = ttk.Button(detail_frame, text="\u2713 Apply & Select Tuning", state="disabled")
        apply_btn.pack(fill="x", pady=4)

        def update_selection(event=None):
            sel = listbox.curselection()
            if not sel:
                title_lbl.config(text="")
                notes_raw_lbl.config(text="")
                strings_lbl.config(text="")
                apply_btn.config(state="disabled")
                return
            tuning_name = names[sel[0]]
            notes_str = tunings[tuning_name]
            title_lbl.config(text=tuning_name)
            notes_raw_lbl.config(text=f"Formula: {notes_str}")
            try:
                parsed = infer_string_notes(notes_str)
                lines = [f"• {p[0]}: {p[2]:.2f} Hz" for p in parsed]
                strings_lbl.config(text="\n".join(lines))
                apply_btn.config(state="normal", command=lambda: self._apply_and_close(tuning_name))
            except Exception as e:
                strings_lbl.config(text=f"Error parsing notes: {e}")
                apply_btn.config(state="disabled")

        listbox.bind("<<ListboxSelect>>", update_selection)
        listbox.bind("<Double-Button-1>", lambda e: apply_btn.invoke() if str(apply_btn["state"]) == "normal" else None)

        if names:
            # Trigger initial preview
            update_selection()

    def _build_custom_tab(self, parent: ttk.Frame):
        container = ttk.Frame(parent, padding=8)
        container.pack(fill="both", expand=True)

        form = ttk.Frame(container)
        form.pack(fill="x", pady=(0, 12))

        ttk.Label(form, text="Tuning Name:").grid(row=0, column=0, sticky="w", pady=4)
        name_var = tk.StringVar(value="Custom Tuning")
        name_entry = ttk.Entry(form, textvariable=name_var, width=32)
        name_entry.grid(row=0, column=1, sticky="w", padx=8, pady=4)

        ttk.Label(form, text="Category:").grid(row=1, column=0, sticky="w", pady=4)
        existing_cats = list(self.tunings_mgr.get_categories().keys())
        cat_var = tk.StringVar(value=existing_cats[0] if existing_cats else "6-String")
        cat_combo = ttk.Combobox(form, textvariable=cat_var, values=existing_cats + ["Custom"], width=30)
        cat_combo.grid(row=1, column=1, sticky="w", padx=8, pady=4)

        ttk.Label(form, text="Notes:").grid(row=2, column=0, sticky="w", pady=4)
        notes_var = tk.StringVar(value="D A D G B E")
        notes_entry = ttk.Entry(form, textvariable=notes_var, width=32)
        notes_entry.grid(row=2, column=1, sticky="w", padx=8, pady=4)

        hint = ttk.Label(
            form,
            text="Space-separated notes from low to high string.\nOctave is optional (e.g. 'C G C F A D' or 'C2 G2 C3 F3 A3 D4').",
            foreground="#666666",
            font=("Helvetica", 9)
        )
        hint.grid(row=3, column=0, columnspan=2, sticky="w", pady=(2, 8))

        preview_box = ttk.LabelFrame(container, text="Live Inferred Strings Preview", padding=10)
        preview_box.pack(fill="both", expand=True, pady=(0, 12))

        preview_lbl = ttk.Label(preview_box, text="", font=("Helvetica", 10), justify="left")
        preview_lbl.pack(anchor="nw", fill="both", expand=True)

        status_lbl = ttk.Label(container, text="", foreground="#c0392b")
        status_lbl.pack(anchor="w", pady=(0, 4))

        def refresh_preview(*args):
            text = notes_var.get().strip()
            if not text:
                preview_lbl.config(text="(Enter note names above)")
                status_lbl.config(text="")
                return
            try:
                parsed = infer_string_notes(text)
                if not parsed:
                    preview_lbl.config(text="No notes recognized.")
                    status_lbl.config(text="Please enter at least one note.")
                    return
                chips = "   ".join([f"{p[0]} ({p[2]:.1f} Hz)" for p in parsed])
                preview_lbl.config(text=f"Total Strings: {len(parsed)}\n\n{chips}")
                status_lbl.config(text="")
            except Exception as e:
                preview_lbl.config(text="")
                status_lbl.config(text=f"Invalid format: {e}")

        notes_var.trace_add("write", refresh_preview)
        refresh_preview()

        def save_and_apply():
            t_name = name_var.get().strip()
            t_cat = cat_var.get().strip()
            t_notes = notes_var.get().strip()
            if not t_name:
                messagebox.showerror("Validation Error", "Please provide a name for this tuning.", parent=self)
                return
            if not t_cat:
                t_cat = "6-String"

            ok = self.tunings_mgr.add_custom_tuning(t_cat, t_name, t_notes)
            if ok:
                self.on_applied(t_name)
                self.destroy()
            else:
                messagebox.showerror("Error", "Could not parse or save tuning. Please check notes.", parent=self)

        ttk.Button(container, text="\u2713 Save & Apply Tuning", command=save_and_apply).pack(fill="x")

    def _apply_and_close(self, tuning_name: str):
        if self.tunings_mgr.set_active_tuning(tuning_name):
            self.on_applied(tuning_name)
        self.destroy()
