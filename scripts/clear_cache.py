#!/usr/bin/env python3
"""Clear all caches to start fresh experiment."""

import os
import shutil
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))


def clear_all_caches():
    """Clear all cached data and indexes."""
    print("🗑️  Clearing all caches...")

    # Clear knowledge base cache
    cache_file = "knowledge_base_cache.json"
    if os.path.exists(cache_file):
        os.remove(cache_file)
        print(f"✅ Removed {cache_file}")

    # Clear ChromaDB persistence directory
    chroma_dir = "chroma_db"
    if os.path.exists(chroma_dir):
        shutil.rmtree(chroma_dir)
        print(f"✅ Removed {chroma_dir}/")

    # Clear results directory (optional)
    results_dir = "results"
    if os.path.exists(results_dir):
        response = input(f"Clear results directory '{results_dir}'? (y/N): ")
        if response.lower().startswith("y"):
            shutil.rmtree(results_dir)
            print(f"✅ Removed {results_dir}/")
        else:
            print(f"📁 Keeping {results_dir}/")

    print("\n🎉 All caches cleared! Next run will create fresh data.")
    print("\n💡 This will cause:")
    print("   • New random sample of 10 questions + 50 documents")
    print("   • Fresh vector index creation")
    print("   • Longer first run (but consistent subsequent runs)")


def main():
    """Main function."""
    print("RAG Cache Cleaner")
    print("=" * 30)

    # Show current cache status
    cache_file = "knowledge_base_cache.json"
    chroma_dir = "chroma_db"

    print("📋 Current cache status:")
    print(
        f"   • Knowledge base cache: {'✅ EXISTS' if os.path.exists(cache_file) else '❌ MISSING'}"
    )
    print(
        f"   • ChromaDB index: {'✅ EXISTS' if os.path.exists(chroma_dir) else '❌ MISSING'}"
    )

    if not os.path.exists(cache_file) and not os.path.exists(chroma_dir):
        print("\n✨ No caches found - nothing to clear!")
        return

    print("\n⚠️  WARNING: This will clear all cached data!")
    print("   You'll get different questions/documents on next run.")

    response = input("\nProceed with cache clearing? (y/N): ")
    if response.lower().startswith("y"):
        clear_all_caches()
    else:
        print("❌ Cache clearing cancelled")


if __name__ == "__main__":
    main()
