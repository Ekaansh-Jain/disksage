# Security and safety policy

disksage deletes files, so "it offered to delete something it shouldn't have" is treated as a security issue.

## Report privately

Open a private report at <https://github.com/Ekaansh-Jain/disksage/security/advisories/new>. Please include:

- your disksage version (`disksage --version`) and macOS version
- the path, with `~` for your home folder, and the label and tier disksage gave it
- whether it was pre-ticked, and whether anything was actually deleted

I aim to reply within 3 days, and to release a fix within 7 days for anything that can lose data.

## What counts

- A path offered for deletion that holds real data: documents, photos, app data, credentials, source code
- A pre-ticked item that doesn't rebuild itself
- Any way to make the AI mark something SAFE
- Any way to send folder paths to an AI provider you didn't choose
- Anything outside your home folder being touched
- A crash that leaves files half-moved or half-cloned

## What doesn't

- Sizes that look wrong: open a normal bug report
- Folders disksage doesn't recognize: use the "Unrecognized or mislabelled folder" form

## What disksage guarantees

- It only acts inside your home folder.
- It never offers keychains, app containers, Mail, Messages, Photos and other libraries, iCloud Drive, cloud-sync folders, iPhone backups, credential folders, or anything inside a macOS package.
- Only items from the built-in list can be SAFE. The AI can label items, never mark them SAFE.
- Folder paths and sizes go only to the AI provider you chose. With "No AI", nothing is sent anywhere.
- Caches that rebuild themselves are deleted after you confirm; everything else goes to the Trash.
- Duplicate models are replaced with APFS clones only after their contents are verified to be identical.
- Folders with git-tracked files or unpushed commits are flagged and never pre-ticked.
