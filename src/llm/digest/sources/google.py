"""Google Workspace (Calendar, Gmail, Drive) source ingestor."""

from __future__ import annotations

import datetime as dt
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from llm.digest.cache import ActivityCache
    from llm.digest.filter import DigestFilter
    from llm.settings.models import GoogleSettings

SCOPES = [
    "https://www.googleapis.com/auth/calendar.readonly",
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/drive.readonly",
    "https://www.googleapis.com/auth/drive.metadata.readonly",
]


def _get_google_credentials(settings: GoogleSettings) -> Any:
    """Load cached OAuth credentials or None."""
    token_file = settings.resolved_token_path
    if not token_file.exists():
        return None

    try:
        from google.oauth2.credentials import Credentials  # noqa: PLC0415

        creds = Credentials.from_authorized_user_file(str(token_file), SCOPES)
        return creds
    except Exception:
        return None


def ingest_google(
    settings: GoogleSettings,
    start_time: dt.datetime,
    end_time: dt.datetime,
    digest_filter: DigestFilter | None = None,
    cache: ActivityCache | None = None,
) -> list[dict[str, Any]]:
    """Ingest Calendar events, Gmail priority unreads, and modified Drive files."""
    creds = _get_google_credentials(settings)
    if not creds:
        return []

    items: list[dict[str, Any]] = []

    # 1. Google Calendar
    try:
        from googleapiclient.discovery import build  # noqa: PLC0415

        cal_service = build("calendar", "v3", credentials=creds)
        # Fetch events for the target day window
        events_result = (
            cal_service.events()
            .list(
                calendarId=settings.calendar_id,
                timeMin=start_time.isoformat(),
                timeMax=end_time.isoformat(),
                singleEvents=True,
                orderBy="startTime",
            )
            .execute()
        )
        events = events_result.get("items", [])
        for ev in events:
            summary = ev.get("summary", "Untitled Meeting")
            start = ev.get("start", {}).get("dateTime", ev.get("start", {}).get("date"))
            desc = ev.get("description", "")
            location = ev.get("location", "")
            hangout_link = ev.get("hangoutLink", "")
            event_id = ev.get("id", "")

            content = (
                f"Time: {start}\nLocation: {location}\nMeet: {hangout_link}\nDescription: {desc}".strip()
            )
            iid = f"google_calendar_{event_id}"
            item = {
                "id": iid,
                "source": "google_calendar",
                "item_id": event_id,
                "thread_id": f"calendar_{event_id}",
                "title": f"Calendar: {summary}",
                "author": ev.get("organizer", {}).get("email", ""),
                "content": content,
                "timestamp": start,
                "url": ev.get("htmlLink", ""),
                "metadata": {"type": "calendar_event"},
            }
            items.append(item)
            if cache:
                cache.upsert_item(**item)
    except Exception:
        pass

    # 2. Gmail Priority Unreads
    try:
        from googleapiclient.discovery import build  # noqa: PLC0415

        gmail_service = build("gmail", "v1", credentials=creds)
        query = settings.gmail_query_filters
        res = gmail_service.users().messages().list(userId="me", q=query, maxResults=15).execute()
        messages = res.get("messages", [])

        for msg_meta in messages:
            msg_id = msg_meta.get("id")
            msg = (
                gmail_service.users()
                .messages()
                .get(userId="me", id=msg_id, format="metadata", metadataHeaders=["Subject", "From", "Date"])
                .execute()
            )
            headers_list = msg.get("payload", {}).get("headers", [])
            headers_dict = {h["name"].lower(): h["value"] for h in headers_list}

            subject = headers_dict.get("subject", "No Subject")
            from_addr = headers_dict.get("from", "")
            date_str = headers_dict.get("date", "")
            snippet = msg.get("snippet", "")

            if digest_filter and digest_filter.should_ignore_author(from_addr):
                continue

            iid = f"gmail_{msg_id}"
            item = {
                "id": iid,
                "source": "gmail",
                "item_id": msg_id,
                "thread_id": f"gmail_thread_{msg.get('threadId', msg_id)}",
                "title": f"Email: {subject}",
                "author": from_addr,
                "content": snippet,
                "timestamp": date_str,
                "url": f"https://mail.google.com/mail/u/0/#inbox/{msg_id}",
                "metadata": {"snippet": snippet},
            }
            items.append(item)
            if cache:
                cache.upsert_item(**item)
    except Exception:
        pass

    # 3. Google Drive recently modified
    try:
        from googleapiclient.discovery import build  # noqa: PLC0415

        drive_service = build("drive", "v3", credentials=creds)
        time_min_str = start_time.strftime("%Y-%m-%dT%H:%M:%SZ")
        q = f"modifiedTime >= '{time_min_str}' and trashed = false"
        files_res = (
            drive_service.files()
            .list(
                q=q,
                fields="files(id, name, modifiedTime, webViewLink, owners)",
                pageSize=20,
            )
            .execute()
        )
        files = files_res.get("files", [])
        for f in files:
            file_id = f.get("id")
            name = f.get("name")
            mod_time = f.get("modifiedTime")
            owners = f.get("owners", [])
            owner_name = owners[0].get("displayName", "") if owners else ""

            iid = f"google_drive_{file_id}"
            item = {
                "id": iid,
                "source": "google_drive",
                "item_id": file_id,
                "thread_id": f"drive_{file_id}",
                "title": f"Google Doc: {name}",
                "author": owner_name,
                "content": f"Modified at: {mod_time}\nOwner: {owner_name}",
                "timestamp": mod_time,
                "url": f.get("webViewLink", ""),
                "metadata": {"type": "drive_file"},
            }
            items.append(item)
            if cache:
                cache.upsert_item(**item)
    except Exception:
        pass

    return items


def search_google_drive(settings: GoogleSettings, query: str, limit: int = 10) -> list[dict[str, Any]]:
    """Search Google Drive by file name or fullText query."""
    creds = _get_google_credentials(settings)
    if not creds:
        return []

    try:
        from googleapiclient.discovery import build  # noqa: PLC0415

        drive_service = build("drive", "v3", credentials=creds)
        safe_q = f"name contains '{query}' or fullText contains '{query}'"
        res = (
            drive_service.files()
            .list(
                q=safe_q,
                fields="files(id, name, modifiedTime, webViewLink, owners, description)",
                pageSize=limit,
            )
            .execute()
        )
        files = res.get("files", [])
        out = []
        for f in files:
            owners = f.get("owners", [])
            owner = owners[0].get("displayName", "") if owners else ""
            out.append(
                {
                    "id": f.get("id"),
                    "name": f.get("name"),
                    "modifiedTime": f.get("modifiedTime"),
                    "owner": owner,
                    "url": f.get("webViewLink"),
                    "description": f.get("description", ""),
                }
            )
        return out
    except Exception:
        return []


def get_drive_file_activity(settings: GoogleSettings, file_id: str) -> dict[str, Any]:
    """Retrieve revision and comment activity for a specific Google Drive file."""
    creds = _get_google_credentials(settings)
    if not creds:
        return {}

    try:
        from googleapiclient.discovery import build  # noqa: PLC0415

        drive_service = build("drive", "v3", credentials=creds)
        file_meta = (
            drive_service.files()
            .get(
                fileId=file_id,
                fields="id, name, modifiedTime, webViewLink, owners, description",
            )
            .execute()
        )
        revs = (
            drive_service.revisions()
            .list(
                fileId=file_id,
                fields="revisions(id, modifiedTime, lastModifyingUser)",
                pageSize=10,
            )
            .execute()
        )
        return {
            "file": file_meta,
            "revisions": revs.get("revisions", []),
        }
    except Exception:
        return {}
